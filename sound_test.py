from os.path import dirname
from os.path import join as join_path

import numpy as np
import simpleaudio as sa
import wave

import matplotlib.pyplot as plt

from multiprocessing import Process, Event, Value
from threading import Thread

from time import perf_counter_ns as timer
from keyboard import add_hotkey, remove_hotkey

from collections.abc import Iterable


USE_24_BIT = True
GAUSSIAN = True


def BASE_WAVEFORM(ts, **kwarks):
    choice = kwarks.get('profile', 1)
    if choice == 2: # Square
        return np.sign(np.sin(2*np.pi*ts*kwarks.get('frequency', 440)))
    elif choice == 3: # Sawtooth
        ret = np.mod(ts, 1/kwarks.get('frequency', 440))
        return ret / np.max(np.abs(ret))-0.5
    elif choice == 4:
        fp = dirname(__file__)
        with wave.Wave_read(join_path(fp,'test.wav')) as wav:
            bit_depth = wav.getsampwidth()
            if bit_depth == 2:
                USE_24_BIT = False
            elif bit_depth == 3:
                USE_24_BIT = True
            pass
    else: # If choice is not 2 or 3 the selected profile is sine wave.
        return np.sin(2*np.pi*ts*kwarks.get('frequency', 440))

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
    # return (ts//(kwarks['duration']/10))/1000 # Monotone step function
    
    return ts/1000 # Linear function

DETECT_KEY = 'space'

def ask(s, t, d=None):
    while 1:
        inp = input(s+('' if d is None else (' (Default: %s)' % d))+': ').replace(',','.')
        try:
            return t(inp if d is None else (inp or d))
        except ValueError:
            print('The given value needs to be a %s!' % t.__name__)

def setup():
    settings = {}

    settings['profile'] = ask('Give the sound profile [1=Sine, 2=Square, 3=Saw, 4=test.wav]', int, d=1)
    if settings['profile'] != 4:
        settings['frequency'] = ask('Give the note frequency [Hz]', float, d=440)
        settings['sample_rate'] = ask('Give the sampling rate [Hz]', int, d=44100)
        settings['duration'] = ask('Give the desired test duration [s]', float, d=5)
    else:
        fp = dirname(__file__)
        with wave.Wave_read(join_path(fp,'test.wav')) as wav:
            settings['sample_rate'] = wav.getframerate()
            settings['duration'] = wav.getnframes()/settings['sample_rate']
        
    
    return settings

def generate_noise(ts, noiseProfile, **kwarks):
    nps = noiseProfile(ts, **kwarks)
    if GAUSSIAN:
        return np.random.normal(0,nps) # Generates random values using gaussian profile (68.2 % chance that the random value is within (+-)nps)
    else:
        return nps*(2*np.random.sample(size=len(nps))-1) # Generates random values linearly between [-nps, nps)

def generate_waveform(pure, noiseProfile, **kwarks):
    d = kwarks.get('duration', 3)
    N = int(d * kwarks.get('sample_rate', 44100))
    
    ts = np.linspace(0, d, N, False)
    
    p = pure(ts, **kwarks)
    n = generate_noise(ts, noiseProfile, **kwarks)
    audio = p + n
    # audio = pure(np.linspace(0, d, N, False), **kwarks)
    
    if USE_24_BIT:
        # This is for 24-bit audio. In sa.WaveObject(...) use bytes_per_sample=3
        audio *= (2**(23) - 1) / np.max(np.abs(audio))
        yield bytearray([b for i,b in enumerate(audio.astype(np.int32).tobytes()) if i % 4 != 3])
    else:
        # This is for 16-bit audio. In sa.WaveObject(...) use bytes_per_sample=2
        audio *= (2**15 - 1) / np.max(np.abs(audio))
        yield audio.astype(np.int16)

    yield ts, p, n


# Threading implementation gave few hundred milliseconds too low value each time i tested it with time.sleep,
#  so i switched to multiprocessing implementation.
# The mainloop implementation also gave some hundred milliseconds too slow values when tested with metronome.
class StopWatch(Process):
    def __init__(self, hotkey=None):
        super().__init__(name='Stop Watch')
        self._stop_event = Event()
        self._stop_event.clear()
        
        self._waiter = Event()
        self._waiter.clear()
        
        self._elapsed_time = Value('Q',0)
        self._hotkey = hotkey or 'space'

        super().start()
        # Waits till the process is started.
        self._stop_event.wait()
    
    def start(self):
        if self.is_alive:
            self._stop_event.set()
        else:
            raise RuntimeError()
    
    def join(self, *args, forceStop=False, **kwarks):
        zeroTimer = not self._waiter.is_set()
        if forceStop and self.is_alive():
            self._waiter.set()
        super().join(*args, **kwarks)
        if zeroTimer:
            with self._elapsed_time.get_lock():
                self._elapsed_time.value = 0
    
    def getValue(self):
        return self._elapsed_time.value or None
    
    def run(self):
        s,e = None, None
        
        def hotkey_action():
            if s is not None:
                self._waiter.set()
            
        remove = add_hotkey(self._hotkey, hotkey_action)
        # Inform the main process that this process is started
        self._stop_event.set()
        # Clears the stop event so that process halts till it is started again with start function.
        self._stop_event.clear()
        self._stop_event.wait()
        
        s = timer()
        self._waiter.wait()
        e = timer()
        
        with self._elapsed_time.get_lock():
            self._elapsed_time.value = e-s
        remove_hotkey(remove)

def init_stopwatch(res, i):
    res[i] = StopWatch(hotkey=DETECT_KEY)


def measure(settings, SW):
    print('\nGenerating the sound sample', end='...')
    wf, raw  = generate_waveform(BASE_WAVEFORM, NOISE_PROFILE, **settings)
    wave_obj = sa.WaveObject(wf, num_channels=1, bytes_per_sample=3 if USE_24_BIT else 2, sample_rate=settings.get('sample_rate', 44100))
    print('Done!')
    
    print('\nPress \'%s\' when you hear the white noise.' % DETECT_KEY)
    input('Press enter to start the experiment!')
    
    # s = timer()
    SW.start()
    play_obj = wave_obj.play()
    remove = add_hotkey(DETECT_KEY, play_obj.stop)
    
    play_obj.wait_done()
    # e = timer()
    remove_hotkey(remove)
    SW.join(forceStop=True)
    
    
    # t = (e-s)/1e9
    # if t > settings.get('duration', 10):
    #     print('You didn\'t detect the white noise!')
    #     t=None
    # else:
    #     print('You detected the white noise at %.3f s.' % t)
        
    t = None
    if SW.getValue() is None:
        print('You didn\'t detect any white noise!')
    else:
        t = SW.getValue()/1e9
        print('You detected noise at %.3f s.' % t)
    
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
        
        # # Mark the detected spot with vertical red line or...
        # t_SW = SW.getValue()/1e9
        # ax2.axvline(t_SW, color='blue')
        # ax1.axvline(t_SW, color='blue')
        # ax.axvline(t_SW, color='blue', label=('%.3f dB (@ %.3f s).' % (y, t)))
        
        # # ...black cross.
        # ax.scatter(t, y, label=('%.3f dB (@ %.3f s).' % (y, t)),zorder=1, c='k', marker='x')
        
        # Don't draw legend for the detected spot if nothing was detected.
        ax.legend(loc=2)
    
    ax.grid()
    ax2.set_xlabel('Time [s]')
    ax.set_ylabel('$\mathrm{SNR_{dB}}$')
    
    
    plt.show()

def main():
    # The stopwatch process will be started in parallel with settings inquiry from user in a separate thread because starting process will take noticable amount of time otherwise.
    result = [None]
    init_stopwatch_thread = Thread(name='Stopwatch init', target=init_stopwatch, args=(result, 0))
    init_stopwatch_thread.start()
    
    settings = setup()
    
    init_stopwatch_thread.join()
    SW = result[0]
    del result
    
    try:
        measurement = measure(settings, SW)
        plot(*measurement, settings)
    except Exception as e:
        # If exception occurs for any reason, try stopping the stopwatch process.
        SW.join(forceStop=True, timeout=1)
        raise e


if __name__ == "__main__":
    main()