import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

plt.style.use('seaborn-v0_8-paper')

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

c = 3e8

# =========================
# (TODAS TUS FUNCIONES IGUAL - NO CAMBIO NADA)
# =========================

def rayleigh(fc, v, Nwaves=512):
    fm = v*fc/c
    fs = 20*fm
    t = np.arange(0, 5/fm, 1/fs)

    beta = np.pi*np.arange(1,Nwaves+1)/Nwaves
    phi  = 2*np.pi*np.random.rand(Nwaves)

    cos_term = np.cos(2*np.pi*fm*np.cos(beta)[:,None]*t + phi[:,None])
    sin_term = np.sin(2*np.pi*fm*np.cos(beta)[:,None]*t + phi[:,None])

    h = (np.sum(cos_term,0) + 1j*np.sum(sin_term,0))/np.sqrt(Nwaves)
    h /= np.sqrt(np.mean(np.abs(h)**2))

    return t, h, fm

def rician(fc, v, K_dB, Nwaves=512):
    t, h_ray, fm = rayleigh(fc, v, Nwaves)
    K = 10**(K_dB/10)
    h_los = np.exp(1j*2*np.pi*fm*t)
    h = np.sqrt(K/(K+1))*h_los + np.sqrt(1/(K+1))*h_ray
    h /= np.sqrt(np.mean(np.abs(h)**2))
    return t, h, fm

def rician_from_rayleigh(h_ray, fm, t, K_dB):
    K = 10**(K_dB/10)
    if K == 0:
        return h_ray 
    h_los = np.exp(1j*2*np.pi*fm*t)
    h = np.sqrt(K/(K+1))*h_los + np.sqrt(1/(K+1))*h_ray
    h /= np.sqrt(np.mean(np.abs(h)**2))
    return h

def multipath_channel(fc, v, d, use_rician=False, K_dB=10):

    delays_us = np.array([0, 0.2, 0.8, 1.2, 2.3, 3.7])
    powers_dB = np.array([0,-0.9,-4.9,-8.0,-7.8,-23.9])

    delays = delays_us*1e-6
    powers = 10**(powers_dB/10)
    powers = powers / np.sum(powers)

    PL = path_loss(d) * shadowing()
    paths=[]

    for k in range(len(delays)):
        if k == 0 and use_rician:
            t,h,fm = rician(fc, v, K_dB)
        else:
            t,h,fm = rayleigh(fc, v)

        paths.append(np.sqrt(powers[k]) * h * PL)

    return t, np.array(paths), delays, fm

def frequency_response(paths,delays,BW=300e3,Nf=4096):
    L,Nt = paths.shape
    f = np.linspace(-BW,BW,Nf)

    H = np.zeros((Nt,Nf),dtype=complex)
    for k in range(L):
        H += paths[k,:,None]*np.exp(-1j*2*np.pi*f*delays[k])

    return f,H

def delay_dispersion(delays,power_paths,fm):
    P = power_paths/np.sum(power_paths)
    tau_mean = np.sum(P*delays)
    sigma_tau = np.sqrt(np.sum(P*(delays-tau_mean)**2))
    Bc = 1/(sigma_tau*5)
    Tc = 1/fm
    return tau_mean,sigma_tau,Bc,Tc

def bandlimited_impulse(delays, power_paths, BW):
    tau_min = np.min(delays)
    tau_max = np.max(delays)
    margin = 5 / BW

    tau = np.linspace(tau_min - margin, tau_max + margin, 4000)
    h_tau = np.zeros_like(tau)

    sigma = 1 / BW

    for k in range(len(delays)):
        ak = np.sqrt(power_paths[k])
        window = np.exp(-(tau - delays[k])**2 / (2 * sigma**2))
        h_tau += ak * np.sinc(BW * (tau - delays[k])) * window

    return tau, h_tau

def path_loss(d, d0=1, n=3):
    return np.sqrt((d0/d)**n)

def shadowing(sigma_dB=6):
    return 10**(np.random.normal(0, sigma_dB)/20)

# =========================
# MAIN (SIN CAMBIOS)
# =========================
def ejecutar():

    mode = int(combo.get())
    d = float(entry_d.get())
    fc = float(entry_fc.get())
    v  = float(entry_v.get())

    use_rician = (mode == 2)
    K_dB = float(entry_k.get()) if use_rician else None

    t, paths, delays, fm = multipath_channel(fc, v, d, use_rician, K_dB)

    t_cmp, h_ray_base, fm = rayleigh(fc, v)

    if use_rician:
        h_cmp = rician_from_rayleigh(h_ray_base, fm, t_cmp, K_dB)
    else:
        h_cmp = h_ray_base.copy()

    power_paths = np.mean(np.abs(paths)**2,axis=1)
    tau_mean,sigma_tau,Bc,Tc = delay_dispersion(delays,power_paths,fm)

    f_axis,H = frequency_response(paths,delays,Bc*2)
    idx=len(t)//2

    tau,h_tau = bandlimited_impulse(delays,power_paths,Bc*10)

    BW_flat = Bc / 10
    BW_sel  = Bc * 10

    f_flat, H_flat = frequency_response(paths, delays, BW_flat)
    f_sel,  H_sel  = frequency_response(paths, delays, BW_sel)

    # =========================
    # FIGURA 1 (TUYA EXACTA)
    # =========================
    fig = plt.figure(figsize=(12,10))

    plt.subplot(2,2,1)
    plt.plot(f_axis/1e6,20*np.log10(np.abs(H[idx])))
    plt.title("Respuesta en Frecuencia del Canal")
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("|H(f)| dB")
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(tau*1e6,np.abs(h_tau),'g')
    for k in range(len(delays)):
        amp=np.sqrt(power_paths[k])
        plt.plot(delays[k]*1e6,amp,'o')
        plt.vlines(delays[k]*1e6,0,amp)
    plt.title("Bandlimited impulse response")
    plt.xlabel("Delay (µs)")
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(t_cmp*1000,20*np.log10(np.abs(h_ray_base)),
             label="Rayleigh (misma realización)", alpha=0.7)
    plt.plot(t_cmp*1000,20*np.log10(np.abs(h_cmp)),
             label=f"Rician (misma realización, K={K_dB} dB)", linewidth=2)
    plt.title("Comparación fading temporal (misma trayectoria)")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencia (dB)")
    plt.legend()
    plt.grid()

    plt.subplot(2,2,4)
    if use_rician:
        plt.plot(t_cmp*1000, 20*np.log10(np.abs(h_cmp)),
                 label=f"Rician (K={K_dB} dB)", linewidth=2)
        plt.title("Fading temporal (Rician)")
    else:
        plt.plot(t_cmp*1000, 20*np.log10(np.abs(h_ray_base)),
                 label="Rayleigh", linewidth=2)
        plt.title("Fading temporal (Rayleigh)")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Potencia (dB)")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # =========================
    # FIGURA 2 (TUYA EXACTA)
    # =========================
    plt.figure(figsize=(8,8))

    plt.subplot(2,1,1)
    plt.plot(f_flat/1e3, 20*np.log10(np.abs(H_flat[idx])))
    plt.title("Flat Fading (BW << Bc)")
    plt.ylabel("|H(f)| dB")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(f_sel/1e3, 20*np.log10(np.abs(H_sel[idx])))
    plt.title("Frequency Selective (BW >> Bc)")
    plt.xlabel("Frecuencia (kHz)")
    plt.ylabel("|H(f)| dB")
    plt.grid()


    texto = f"""
    --- RESULTADOS ---

    fm = {fm:.2f} Hz

    Delay medio (tau_mean) = {tau_mean*1e6:.3f} us
    RMS delay spread (sigma_tau) = {sigma_tau*1e6:.3f} us

    Ancho de banda de coherencia (Bc) = {Bc/1e3:.3f} kHz
    Tiempo de coherencia (Tc) = {Tc*1e3:.3f} ms
    """

    resultados.config(text=texto)



    plt.show()

# =========================
# INTERFAZ
# =========================
root = tk.Tk()
root.title("Simulador de Canal")

tk.Label(root, text="1 Rayleigh | 2 Rician").pack()
combo = ttk.Combobox(root, values=["1","2"])
combo.current(0)
combo.pack()

tk.Label(root, text="Distancia (m)").pack()
entry_d = tk.Entry(root)
entry_d.pack()

tk.Label(root, text="fc (Hz)").pack()
entry_fc = tk.Entry(root)
entry_fc.pack()

tk.Label(root, text="v (m/s)").pack()
entry_v = tk.Entry(root)
entry_v.pack()

tk.Label(root, text="K (dB)").pack()
entry_k = tk.Entry(root)
entry_k.pack()

tk.Button(root, text="Simular", command=ejecutar).pack()

resultados = tk.Label(root, text="", justify="left", font=("Consolas", 10))
resultados.pack(pady=10)

root.mainloop()