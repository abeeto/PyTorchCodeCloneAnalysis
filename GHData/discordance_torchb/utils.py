from music21 import midi, environment
import base64
import numpy as np
from tqdm import tqdm


environment.UserSettings()['warnings'] = 0

BEAT_DIV = 32
TICKS_PER_Q = 192
PERC_MAP = {
    # KICK GROUP
    0:  36,
    # SNARE AND RIMS
    1:  40,
    2:  37,
    # TOMS
    3:  41,  # low
    4:  47,  # mid
    5:  50,  # high
    # WORLD
    6:  64,  # low percs african
    7:  63,  # high percs african
    8:  68,  # latin a
    9:  77,  # latin b
    10: 56,  # unusual, unique
    # HH / CYMB
    11: 42,  # muted hh
    12: 46,  # open hat / splash
    13: 49,  # crash and chinese
    14: 51  # rides
}
PERC_GROUPS = [
    [0],
    [1, 2],
    [3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14]
]
EIGHT_PERC_GROUPS = [
    [0],
    [1],
    [2, 3, 4],
    [5, 6, 7]
]

EIGHT_MAP = {
    0: (36, [0]),
    1: (40, [1, 2]),
    2: (41, [3, 4, 6]),
    3: (50, [5, 7, 9]),
    4: (56, [10, 8]),
    5: (42, [11]),
    6: (46, [12, 13]),
    7: (51, [14])
}


def map2twelve(seqs):
    """
    maps MxNx20 sequences to MxNx12 sequences
    from 15 percs and 5 vel groups to 8 percs and 4 vel groups
    :param seqs: array of sequences to be converted
    :return: MxNx12 numpy sequence
    """
    def pmap(s, mem):
        nstep = np.zeros(12)
        for k, mm in EIGHT_MAP.iteritems():
            for idx in mm[1]:
                if s[idx] > 0.0 and (k not in mem or mem[k] == idx):
                    nstep[k] = 1.0
                    if k > 2:
                        mem[k] = idx
        # manual vel map
        nstep[8] = s[15]
        nstep[9] = s[16]
        nstep[10] = (s[17] + s[18]) / 2.0
        nstep[11] = s[19]
        return nstep

    mapped = []
    for i, seq in enumerate(tqdm(seqs, 'map to 8')):
        nseq = []
        mem = {}
        for j, step in enumerate(seq):
            nstep = pmap(step, mem)
            nseq.append(nstep)
        mapped.append(nseq)
    return np.array(mapped)




def normalizer(seq):
    """
    compress / normalize velocities
    """
    vels = seq[:, -4:]
    for i, vel in enumerate(vels.T):
        med = vel[np.nonzero(vel)].mean()
        gaps = med - vel[np.nonzero(vel)]
        vel[np.nonzero(vel)] += (gaps / 1.7)
    trunk = np.delete(seq, np.s_[8::], 1)
    gap = 0.98 - np.max(vels)
    vels[np.nonzero(vels)] = np.absolute(vels[np.nonzero(vels)] + gap)
    trunk = np.column_stack((trunk, vels))
    return trunk


def clean_ml_out(seq):
    """
    Restore a clean version of a sequence that outputed of a neural network
    :param seq:
    :return:
    """
    # find the group
    def fgroup(idx):
        group = None
        for index, grp in enumerate(EIGHT_PERC_GROUPS):
            if idx in grp:
                group = index
        return group
    for j, step in enumerate(seq):
        step_grps = []
        for k, perc in enumerate(step[:8]):
            seq[j][k] = round(perc)
            grp = fgroup(k)
            if seq[j][k] == 1.0:
                step_grps.append(grp)
        if step.shape[0] > 8:
            for i in range(0, 4):
                if i not in step_grps:
                    step[8 + i] = 0
    return seq


def decompress(str, bar):
    """
    decompress zipped string to numpy seq
    """
    zipped = base64.decodestring(str)
    b64 = zipped.decode('zlib')
    arr = np.frombuffer(base64.decodestring(b64))
    rshaped = arr.reshape(len(arr)/bar/20, bar, 20)
    return rshaped


def compress(seq):
    """
    compress a numpy seq to a zipped base64 (to store in DB)
    """
    b64 = base64.b64encode(seq)
    compressed = base64.encodestring(b64.encode('zlib'))
    return compressed


def draw(seq, bar = 64, quarter = 16):
    """
    Draws the seq in the terminal
    :param seq: seq to be drawn
    :return:
    """
    perc_len = 15
    if seq.shape[1] == 12:
        perc_len = 8
    st = ""
    for i in reversed(range(0, len(seq[0]))):
        if i < perc_len:
            for j in range(0, len(seq)):
                if seq[j][i] > 0.0:
                    st += 'X'
                else:
                    char = ''
                    if j % bar == 0:
                        char = '|'
                    elif  j % quarter == 0:
                        char = ';'
                    else:
                        char = '.'
                    st += char
        else:
            for j in range(0, len(seq)):
                if seq[j][i] > 0.75:
                    st += 'O'
                elif seq[j][i] > 0.0:
                    st += 'o'
                else:
                    st += ' '
        st += "\n"
    return st


def np_seq2mid(np_seq):
    """
    Converts a numpy array to a midi file.
    :param np_seq: numpy beat sequence
    :return: music21.midi.MidiFile
    """
    perc_len = 15
    perc_groups = PERC_GROUPS
    perc_map = PERC_MAP

    if np_seq.shape[1] == 12:
        perc_len = 8
        perc_groups = EIGHT_PERC_GROUPS
        perc_map = {}
        for key, value in EIGHT_MAP.iteritems():
            perc_map[key] = value[0]
    mt = midi.MidiTrack(1)
    t = 0
    tlast = 0
    for step in np_seq:
        # onset will be true if at least one trig is > 0.0
        # the remaining trigs are added at the same delta time
        onset = False # we encountered an onset at this step
        for idx, trig in enumerate(step[:perc_len]):
            # find the group
            group = None
            for index, grp in enumerate(perc_groups):
                if idx in grp:
                    group = index
            if trig > 0.0:
                vel = int(step[perc_len+group]*127)
                pitch = perc_map[idx]
                dt = midi.DeltaTime(mt)
                if onset is False:
                    dt.time = t - tlast
                else:
                    dt.time = 0
                mt.events.append(dt)
                me = midi.MidiEvent(mt)
                me.type = "NOTE_ON"
                me.channel = 10
                me.time = None  # d
                me.pitch = pitch
                me.velocity = vel
                mt.events.append(me)
                if onset is False:
                    tlast = t + 6
                    onset = True
        if onset is True:
            # reset onset for the noteoff
            onset = False
            # makes the note off now
            for idx, trig in enumerate(step[:perc_len]):
                if trig > 0.0:
                    pitch = perc_map[idx]
                    dt = midi.DeltaTime(mt)
                    if onset is False:
                        dt.time = 6
                    else:
                        dt.time = 0
                    mt.events.append(dt)
                    me = midi.MidiEvent(mt)
                    me.type = "NOTE_OFF"
                    me.channel = 10
                    me.time = None  # d
                    me.pitch = pitch
                    me.velocity = 0
                    mt.events.append(me)
                    if onset is False:
                        onset = True
        t += TICKS_PER_Q/BEAT_DIV
    # add end of track
    dt = midi.DeltaTime(mt)
    dt.time = 0
    mt.events.append(dt)
    me = midi.MidiEvent(mt)
    me.type = "END_OF_TRACK"
    me.channel = 1
    me.data = ''  # must set data to empty string
    mt.events.append(me)
    # make midi file
    mf = midi.MidiFile()
    mf.ticksPerQuarterNote = TICKS_PER_Q  # cannot use: 10080
    mf.tracks.append(mt)
    return mf

def augment(beats):
        """
        basic data augmentation
        """
        a = beats[:,:64,:]
        b = beats[:,64:,:]
        t1 = np.concatenate((b, a), axis=1)
        a = beats[:,:32,:]
        b = beats[:,32:64,:]
        c = beats[:,64:96,:]
        d = beats[:,64:96,:]
        t2 = np.concatenate((a,d,c,b), axis=1)
        t3 = np.concatenate((c,d,a,b), axis=1)
        return np.concatenate((t1, t2, t3), axis=0)

def clean_and_unique_beats(beats):
    """
    clean beats to have a good dataset before ML algos
    input is in the form of np.array(n x 128 x 20)
    """

    # UNIQUES
    bincopy = beats[:,:,:15]
    # get only uniques
    uniques, idxs = np.unique(bincopy, axis=0, return_index=True)
    beats_uniques = beats[idxs]

    # INTREDASTINGS
    valids = []
    # check thoses bytes
    for b in tqdm(beats_uniques, 'clean beats'):
        b1 = b[:64,:]
        b2 = b[64:,:]
        m1 = np.mean(b1)
        m2 = np.mean(b2)
        # if m1 > 0 and m2 > 0 and b1[0][0] == 1 and np.mean(b) > 0.008:
        if m1 > 0 and m2 > 0 and b1[0][0] == 1 and np.mean(b) > 0.009:
            valids.append(b)

    return np.concatenate((np.array(valids), augment(np.array(valids))), axis=0)
