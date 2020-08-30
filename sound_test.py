import numpy as np
import simpleaudio as sa

import matplotlib.pyplot as plt

from multiprocessing import Process, Event, Value
from threading import Thread

from time import perf_counter_ns as timer
from keyboard import add_hotkey, remove_hotkey

from collections.abc import Iterable


USE_24_BIT = True

# Sine wave
BASE_WAVEFORM = lambda ts, **kwarks: np.sin(2*np.pi*ts*kwarks.get('frequency', 440))


def to_ndarray(f):
    def wrapper(*args, **kwarks):
        ret = f(*args, **kwarks)
        if type(ret) is Iterable and type(ret) is not np.ndarray:
            ret = np.array(ret)
        return ret
    return wrapper

# Linear step function with 10 steps
@to_ndarray
def NOISE_PROFILE(ts, **kwarks):
    return (ts//(kwarks['duration']//2))/10
    

    # return ts/10000

DETECT_KEY = 'space'

def ask(s, t, d=None):
    while 1:
        inp = input(s+('' if d is None else (' (Default: %s)' % d))+': ').replace(',','.')
        try:
            return t(inp if d is None else (inp or d))
        except ValueError:
            print('The note frequency needs to be a %s!' % t.__name__)

def setup():
    settings = {}

    settings['frequency'] = ask('Give the note frequency [Hz]', float, d=440)
    settings['sample_rate'] = ask('Give the sample rate [Hz]', int, d=44100)
    settings['duration'] = ask('Give the desired test duration [s]', float, d=10)
    
    return settings

def generate_noise(ts, noiseProfile, **kwarks):
    nps = noiseProfile(ts, **kwarks)
    return np.random.normal(0,nps) # Generates random values using gaussian profile (68.2 % chance that the random value is within (+-)nps)
    # return nps*(2*np.random.sample(size=len(nps))-1) # Generates random values linearly between [-nps, nps)

def generate_waveform(pure, noiseProfile, **kwarks):
    d = kwarks.get('duration', 3)
    N = int(d * kwarks.get('sample_rate', 44100))
    
    ts = np.linspace(0, d, N, False)
    
    p = pure(ts, **kwarks)
    n = generate_noise(ts, noiseProfile, **kwarks)
    audio = p + n
    # audio = pure(np.linspace(0, d, N, False), **kwarks)
    
    if USE_24_BIT:
        ## This is for 24-bit audio. In sa.WaveObject(...) use bytes_per_sample=3
        audio *= (2**23 - 1) / np.max(np.abs(audio))
        yield bytearray([b for i,b in enumerate(audio.astype(np.int32).tobytes()) if i % 4 != 3])
    else:
        ## This is for 16-bit audio. In sa.WaveObject(...) use bytes_per_sample=2
        audio *= (2**15 - 1) / np.max(np.abs(audio))
        yield audio.astype(np.int16)

    yield ts, p, n



def measure(settings):
    print('\nGenerating the sound sample', end='...')
    wf, raw  = generate_waveform(BASE_WAVEFORM, NOISE_PROFILE, **settings)
    wave_obj = sa.WaveObject(wf, num_channels=2, bytes_per_sample=3 if USE_24_BIT else 2, sample_rate=settings.get('sample_rate', 44100))
    print('Done!')
    
    print('\nPress \'%s\' when you hear the white noise.' % DETECT_KEY)
    input('Press enter to start the experiment!')
    
    play_obj = wave_obj.play()
    s = timer()
    remove = add_hotkey(DETECT_KEY, play_obj.stop)
    
    play_obj.wait_done()
    e = timer()
    remove_hotkey(remove)
    
    t = (e-s)/1e9
    if t > settings.get('duration', 10):
        print('You didn\'t detect the white noise!')
        t=None
    else:
        print('You detected the white noise at %.3f s.' % t)
    
    return wf, raw, t

def plot(wf, raw, t, settings):
    ts, p, n = raw
    fig, (ax,ax1,ax2) = plt.subplots(nrows=3, sharex=True)

    f_np = lambda t: NOISE_PROFILE(t, **settings)
    
    if USE_24_BIT:
        wf_raw = p+n
        wf = (wf_raw * (2**23 - 1) / np.max(np.abs(wf_raw))).astype(np.int32)
    
    
    ax2.scatter(ts,p+n, s=4, zorder=0, label='Waveform')
    ax2.plot(ts,p, zorder=1, c='k', label='Sine wave')
    
    ax2.grid()
    ax2.legend(loc=2)
    
    np_ts = f_np(ts)
    
    # Ignores division errors.
    np.seterr(divide='ignore')
    
    # # Removes zeros
    # np_ts[np_ts==0] = 1e-13
    
    ax1.scatter(ts,n, s=1, zorder=0, label='Noise')
    ax1.plot(ts,np_ts, zorder=1, c='k', linestyle='--', label='Noise profile')
    ax1.plot(ts,-np_ts, zorder=1, c='k', linestyle='--')
    
    ax1.grid()
    ax1.legend(loc=2)
    
    
    ys = 20*np.log10(1/np_ts+1) # == 20 log10(A_signal/A_noise) == 20 log10((p+n)/n); p==1 & n==NOISE_PROFILE
    ax.plot(ts, ys, zorder=0)
    
    if t is not None:
        try:
            y = 20*np.log10(1/f_np(t)+1)
        except ZeroDivisionError:
            y = 0
        
        # Mark the detected spot with vertical red line or...
        ax2.axvline(t, color='red')
        ax1.axvline(t, color='red')
        ax.axvline(t, color='red', label=('%.3f dB (@ %.3f s).' % (y, t)))
        
        # # ...black cross.
        # ax.scatter(t, y, label=('%.3f dB (@ %.3f s).' % (y, t)),zorder=1, c='k', marker='x')
    
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.legend(loc=2)
    
    
    plt.show()

def main():
    settings = setup()
    measurement = measure(settings)
    plot(*measurement, settings)
    


if __name__ == "__main__":
    main()