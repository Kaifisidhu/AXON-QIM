import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function AxonQIMDashboard() {
  const [dose, setDose] = useState(30);
  const [iterations, setIterations] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const runSimulation = async () => {
    setLoading(true);
    try {
      // UPDATE THIS URL TO YOUR LIVE REPLIT BACKEND URL
      const response = await fetch('https://YOUR_REPLIT_URL_HERE/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dose_mj: parseFloat(dose),
          wavelength_nm: 13.5,
          qim_iterations: parseInt(iterations)
        })
      });
      const data = await response.json();
      
      const chartData = data.baseline_cross_section.map((baseVal, index) => ({
        pixel: index,
        StochasticNoise: baseVal,
        AXONCorrected: data.qim_cross_section[index]
      }));
      
      setResults({ ...data, chartData });
    } catch (error) {
      console.error("Connection failed.", error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 border-b border-slate-800 pb-4">
          <h1 className="text-3xl font-bold text-white tracking-wider">AXON-QIM <span className="text-blue-500">ENGINE</span></h1>
          <p className="text-slate-400 text-sm mt-1">Quantum Information Manifold • Holographic Inversion Dashboard</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-slate-900 p-6 rounded-lg border border-slate-800">
            <label className="block text-sm text-slate-400 mb-2">EUV Dose (mJ/cm²)</label>
            <input type="range" min="10" max="80" value={dose} onChange={(e) => setDose(e.target.value)} className="w-full cursor-pointer"/>
            <div className="text-right text-xl font-mono text-blue-400 mt-2">{dose} mJ</div>
          </div>
          
          <div className="bg-slate-900 p-6 rounded-lg border border-slate-800">
            <label className="block text-sm text-slate-400 mb-2">Holographic Iterations (Ω)</label>
            <input type="range" min="1" max="10" value={iterations} onChange={(e) => setIterations(e.target.value)} className="w-full cursor-pointer"/>
            <div className="text-right text-xl font-mono text-blue-400 mt-2">{iterations} Ω</div>
          </div>

          <div className="flex items-center justify-center bg-slate-900 p-6 rounded-lg border border-slate-800">
            <button onClick={runSimulation} disabled={loading} className="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded shadow-[0_0_15px_rgba(37,99,235,0.5)] transition-all disabled:opacity-50">
              {loading ? "INVERTING FIELD..." : "ENGAGE AXON-QIM"}
            </button>
          </div>
        </div>

        {results && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 p-4 rounded border border-slate-800">
                <div className="text-xs text-slate-500 uppercase tracking-widest">Initial Entropy</div>
                <div className="text-2xl font-mono text-red-400">{results.entropy_initial.toFixed(1)} <span className="text-sm">bits</span></div>
              </div>
              <div className="bg-slate-900 p-4 rounded border border-slate-800">
                <div className="text-xs text-slate-500 uppercase tracking-widest">Final Entropy</div>
                <div className="text-2xl font-mono text-green-400">{results.entropy_final.toFixed(1)} <span className="text-sm">bits</span></div>
              </div>
              <div className="bg-slate-900 p-4 rounded border border-slate-800">
                <div className="text-xs text-slate-500 uppercase tracking-widest">Baseline LER</div>
                <div className="text-2xl font-mono text-red-400">{results.ler_baseline_nm.toFixed(2)} <span className="text-sm">nm</span></div>
              </div>
              <div className="bg-slate-900 p-4 rounded border border-slate-800 shadow-[0_0_20px_rgba(34,197,94,0.1)]">
                <div className="text-xs text-slate-500 uppercase tracking-widest">AXON LER</div>
                <div className="text-2xl font-mono text-green-400">{results.ler_qim_nm.toFixed(2)} <span className="text-sm">nm</span></div>
              </div>
            </div>

            <div className="bg-slate-900 p-6 rounded-lg border border-slate-800 h-96">
              <h3 className="text-sm text-slate-400 mb-4 uppercase tracking-widest">Virtual Microscope: Cross-Section Photon Density</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={results.chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="pixel" stroke="#64748b" tick={false} />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }} />
                  <Line type="monotone" dataKey="StochasticNoise" stroke="#ef4444" strokeWidth={1} dot={false} opacity={0.6} />
                  <Line type="monotone" dataKey="AXONCorrected" stroke="#22c55e" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
