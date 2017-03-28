from __future__ import print_function
import numpy as np
import cv2
import midi

import argparse

import sys

parser = argparse.ArgumentParser(description="Parses a synthesia video and creates a midi.")
parser.add_argument('-v', '--verbose', help='verbosely show processor messages', action="store_true")
parser.add_argument('-s', '--stepping', help='enable stepping through all frames', action="store_true")
parser.add_argument('--inspect', help='pause at given frames', nargs='+', type=int, metavar="frame", default=[])
parser.add_argument('--fps', help='input file fps', type=int, default=30)
parser.add_argument('--rounding', help='round notes to intervals', type=int, default=5)
parser.add_argument('-n', '--frames', help='total frames to process', type=int, default=sys.maxint)
parser.add_argument('--octave', help='shift octave', type=int, default=1)
parser.add_argument('--velocity', help='key velocity', type=int, default=100)
parser.add_argument('input', help='input file to be parsed, should be a video recognizable by opencv',
                    type=argparse.FileType('r'))

parser.add_argument('output', help='output file name', type=argparse.FileType('w'))

args = parser.parse_args()

fps = args.fps
verbose = args.verbose
stepping = args.stepping
round_multiple = args.rounding
infile = args.input.name
outfile = args.output.name
frames = args.frames
shift_octave = args.octave
velocity = args.velocity

chords_only = True

track_hues = [100, 40]
inspect_frame = args.inspect

show_keyboard = stepping or inspect_frame != []


def discover_keyboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sort = sorted(contours, cmp=lambda a, b: cmp(a[0][0][0], b[0][0][0]))  # left-most contour
    firstkey_contour = sort[0]
    _, keyboard_y, _, keyboard_h = cv2.boundingRect(firstkey_contour)
    keyboard = frame[keyboard_y:keyboard_y + keyboard_h, 0:frame[0].size]
    return keyboard, keyboard_h, keyboard_y


def process_keys(keyboard):
    gray = cv2.cvtColor(keyboard, cv2.COLOR_BGR2GRAY)

    keys = list()

    def find_keys(white):
        r, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY if white else cv2.THRESH_BINARY_INV)
        thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8))
        contours, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if show_keyboard:
            cv2.drawContours(keyboard, contours, -1, (0, 0, 255) if white else (255, 0, 0), 2)
        for contour in contours:
            moments = cv2.moments(contour)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            if show_keyboard:
                cv2.circle(keyboard, (center_x, center_y), 3, (255, 0, 0), 1)
            keys.append({'white': white, 'point': {'x': center_x, 'y': center_y}})

    find_keys(True)
    find_keys(False)

    if show_keyboard:
        cv2.imshow('keyboard', keyboard)

    keys_sorted = sorted(keys, cmp=lambda key1, key2: cmp(key1['point']['x'], key2['point']['x']))
    return keys_sorted


def process_frames(cap, keyboard_h, keyboard_y, keys):
    frame_data = list()
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if show_keyboard:
            cv2.imshow('video', frame)

        frame_count += 1
        if frame_count >= frames:
            break

        keyframe = frame[keyboard_y:keyboard_y + keyboard_h, 0:frame[0].size]

        hls = cv2.cvtColor(keyframe, cv2.COLOR_BGR2HLS)
        keys_hues = list()
        for key in keys:
            key_x = key['point']['x']
            key_y = key['point']['y']

            hue, l, s = hls[key_y, key_x]

            if l > 230:
                track = -1
            elif l < 40:
                track = -1
            else:
                dist = sorted([(i, abs(th - hue)) for i, th in enumerate(track_hues)],
                              cmp=(lambda d1, d2: cmp(d1[1], d2[1])))
                track = dist[0][0]
                if stepping or frame_count in inspect_frame:
                    cv2.circle(keyframe, (key_x, key_y + 20), 5, (0, 0, 0), 1)
                    cv2.circle(keyframe, (key_x, key_y + 20), 3, np.asfarray(keyframe[key_y, key_x]), 1)

            keys_hues.append(track)

        frame_data.append(keys_hues)

        if verbose:
            print('frame', '{0:03d}'.format(frame_count), ''.join(map(lambda x: str(x+1), keys_hues)))

        if stepping or len(inspect_frame) != 0:
            cv2.imshow('keyboard + presses', keyframe)
            if frame_count in inspect_frame:
                cv2.imshow('keyboard + presses ' + str(frame_count), keyframe)
            delay = 0 if stepping or frame_count in inspect_frame else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                continue
    cv2.destroyAllWindows()
    cap.release()
    return frame_data


def find_first_c(keys):
    WHITE_KEYS = [0, 2, 4, 5, 7, 9, 11]
    BLACK_KEYS = [1, 3, 6, 8, 10]

    def is_c(kn):
        whites_match = all(cond for cond in map(lambda index: keys[kn + index]['white'], WHITE_KEYS))
        blacks_match = all(cond for cond in map(lambda index: not keys[kn + index]['white'], BLACK_KEYS))
        return whites_match and blacks_match

    first_c = -1
    for i, k in enumerate(keys):
        if is_c(i):
            first_c = i
            break
    if first_c == -1:
        raise AssertionError("Could not find first C")
    return first_c


def write_midi(events):
    pattern = midi.Pattern()
    tracks = [midi.Track() for _ in track_hues]
    for track in tracks:
        pattern.append(track)

    pattern.make_ticks_abs()
    for event in events:
        if verbose:
            print("At ", event['when'], "added event", event)
        if event['type'] == "on":
            tracks[event['track']].append(midi.NoteOnEvent(tick=int(event['when']), velocity=velocity, pitch=event['pitch']))
        else:
            tracks[event['track']].append(midi.NoteOffEvent(tick=int(event['when']), pitch=event['pitch']))
    for track in tracks:
        track.append(midi.EndOfTrackEvent(tick=1))
    midi.write_midifile(outfile, pattern)


def create_events(frame_data, pitch_offset):
    def frame_to_tick(f):
        return (float(f) / float(fps)) * 100.0

    # TODO it would probably be better to transpose frame_data and analyze it per key
    state = frame_data[0]
    pressed_when = [None] * len(frame_data[0])
    events = list()
    for frame_number, new_state in enumerate(frame_data[1:]):
        frame_number += 1
        rounded_frame = int(round_multiple * round(float(frame_number) / round_multiple))
        for key_number, new_key_state in enumerate(new_state):
            key_state = state[key_number]
            if new_key_state != key_state:
                pitch = key_number + pitch_offset
                if key_state == -1:  # turn on
                    pressed_when[key_number] = rounded_frame
                    events.append(
                        {'type': 'on', 'track': new_key_state, 'when': frame_to_tick(rounded_frame), 'pitch': pitch})
                else:  # turn off
                    press_started = pressed_when[key_number]
                    duration = rounded_frame - press_started
                    pressed_when[key_number] = None
                    events.append(
                        {'type': 'off', 'track': key_state, 'when': frame_to_tick(rounded_frame), 'pitch': pitch})
                    if verbose:
                        print("Pressed key", key_number, "at", press_started, "till", rounded_frame, "(",
                              duration, ") on track", state[key_number])

                    if new_key_state != -1:  # key switches to different track
                        pressed_when[key_number] = rounded_frame
                        events.append(
                            {'type': 'on', 'track': new_key_state, 'when': frame_to_tick(rounded_frame),
                             'pitch': pitch})

        state = new_state
    return events


def process():
    cap = cv2.VideoCapture(infile)

    ret, frame = cap.read()
    keyboard, keyboard_h, keyboard_y = discover_keyboard(frame)
    keys = process_keys(keyboard)

    frame_data = process_frames(cap, keyboard_h, keyboard_y, keys)

    first_c = find_first_c(keys)

    if verbose:
        print("first c is on position", first_c)

    pitch_offset = 12 * shift_octave + 12 - first_c

    events = create_events(frame_data, pitch_offset)

    def sort_events(e1, e2):
        if e1['when'] == e2['when']:
            if e1['type'] == 'off':
                return -1
            else:
                return 1
        else:
            return cmp(e1['when'], e2['when'])

    events = sorted(events, cmp=sort_events)

    write_midi(events)


process()
