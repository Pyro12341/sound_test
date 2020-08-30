import numpy as np
import simpleaudio as sa

import matplotlib.pyplot as plt

from multiprocessing import Process, Event, Value
from threading import Thread

from time import perf_counter_ns as timer
from keyboard import add_hotkey, remove_hotkey

USE_24_BIT = True

# Sine wave
BASE_WAVEFORM = lambda ts, **kwarks: np.sin(2*np.pi*ts*kwarks.get('frequency', 440))

# Linear step function with 10 steps
def NOISE_PROFILE(ts, **kwarks):
    # n = len(ts)//100
    # return [i//n for i,t in enumerate(ts)]
    return ts/10000


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
    stds = noiseProfile(ts, **kwarks)
    return np.random.normal(0,stds)

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


# Threading implementation gave few hundred milliseconds too low value each time i tested it with time.sleep, so i switched to multiprocessing implementation.
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
    
def main():
    # The stopwatch process will be started in parallel with settings inquiry from user in a separate thread because starting process will take noticable amount of time otherwise.
    result = [None]
    init_stopwatch_thread = Thread(name='Stopwatch init', target=init_stopwatch, args=(result, 0))
    init_stopwatch_thread.start()
    
    settings = setup()
    
    init_stopwatch_thread.join()
    sw = result[0]
    del result
    
    print('\nGenerating the sound sample', end='...')
    wf, (ts, p, n)  = generate_waveform(BASE_WAVEFORM, NOISE_PROFILE, **settings)
    wave_obj = sa.WaveObject(wf, num_channels=2, bytes_per_sample=3 if USE_24_BIT else 2, sample_rate=settings.get('sample_rate', 44100))
    print('Done!')
    
    print('\nPress \'%s\' when you hear the white noise.' % DETECT_KEY)
    input('Press enter to start the experiment!')
    sw.start()
    play_obj = wave_obj.play()
    s = timer()
    remove = add_hotkey(DETECT_KEY, play_obj.stop)
    
    play_obj.wait_done()
    e = timer()
    sw.join(forceStop=True)
    remove_hotkey(remove)
    
    t0 = (e-s)/1e9
    
    print('t0', t0)
    
    if sw.getValue() is None:
        print('You didn\'t detect the white noise!')
        t = None
    else:
        t = sw.getValue()/1e9
        print('You detected noise at %.3f s.' % t)
        
    fig, ax = plt.subplots()

    wf_raw = p+n
        
    if USE_24_BIT:
        wf = (wf_raw * (2**23 - 1) / np.max(np.abs(wf_raw))).astype(np.int32)
    else:
        wf = (wf_raw * (2**15 - 1) / np.max(np.abs(wf_raw))).astype(np.int16)
    
    # ax.scatter(ts,n, zorder=0)
    # ax.plot(ts,NOISE_PROFILE(ts), zorder=1, c='k')
    
    ys = 20*np.log10(1/NOISE_PROFILE(ts)+1)
    ax.plot(ts, ys, zorder=0)
    
    if t is not None:
        y = 20*np.log10(1/NOISE_PROFILE(t)+1)
        ax.scatter(t, y, label=('Noise level detected: %.3f dB (@ %.3f s).' % (y, t)),zorder=1, c='k', marker='x')
        pass
    
    ax.grid()
    ax.set_xlabel('Time [s]')  
    ax.legend(loc=2)
    plt.show()


if __name__ == "__main__":
    main()