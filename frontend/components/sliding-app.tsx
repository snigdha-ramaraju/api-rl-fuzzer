'use client'

import { AnimatePresence, motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Check, FileCode2, GitBranch, Play, ShieldCheck, Upload, WandSparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Brand, DemoLabel } from '@/components/app-shell'
import { NetworkSphere } from '@/components/network-sphere'

type Screen = 'landing' | 'auth' | 'projects' | 'create' | 'configure' | 'testing' | 'results'
const screens: Screen[] = ['landing', 'auth', 'projects', 'create', 'configure', 'testing', 'results']
const labels: Record<Screen, string> = { landing: 'Home', auth: 'Account', projects: 'Projects', create: 'New project', configure: 'Run setup', testing: 'Live testing', results: 'Results' }

export function SlidingApp() {
  const [screen, setScreen] = useState<Screen>('landing')
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login')
  const [runStarted, setRunStarted] = useState(false)
  const [progress, setProgress] = useState(34)

  useEffect(() => {
    const onHash = () => { const value = window.location.hash.slice(1) as Screen; if (screens.includes(value)) setScreen(value) }
    onHash(); window.addEventListener('hashchange', onHash); return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const go = (next: Screen) => { window.location.hash = next; setScreen(next) }
  useEffect(() => { if (!runStarted) return; const timer = setInterval(() => setProgress((value) => Math.min(value + 11, 100)), 900); return () => clearInterval(timer) }, [runStarted])

  return <main className="sliding-app">
    <header className="site-header sliding-header"><button className="brand brand-button" onClick={() => go('landing')}><span className="brand-mark"><span /><span /><span /></span><span>Rhizome</span></button><nav className="slide-nav" aria-label="Application sections">{screens.slice(1).map((item) => <button key={item} className={screen === item ? 'active' : ''} onClick={() => go(item)}>{labels[item]}</button>)}</nav><DemoLabel /></header>
    <div className="slide-progress" aria-label={`Current section: ${labels[screen]}`}><span style={{ width: `${((screens.indexOf(screen) + 1) / screens.length) * 100}%` }} /></div>
    <AnimatePresence mode="wait">
      <motion.section key={screen} className="slide-panel section-shell" initial={{ opacity: 0, x: 60 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -60 }} transition={{ duration: .34, ease: [0.22, 1, .36, 1] }}>
        {screen === 'landing' && <Landing onStart={() => go('auth')} />}
        {screen === 'auth' && <Auth mode={authMode} setMode={setAuthMode} onContinue={() => go('projects')} />}
        {screen === 'projects' && <Projects onNew={() => go('create')} onOpen={() => go('configure')} />}
        {screen === 'create' && <Create onNext={() => go('configure')} />}
        {screen === 'configure' && <Configure onNext={() => { setRunStarted(false); setProgress(34); go('testing') }} />}
        {screen === 'testing' && <Testing progress={progress} started={runStarted} onStart={() => setRunStarted(true)} onResults={() => go('results')} />}
        {screen === 'results' && <Results onBack={() => go('testing')} />}
      </motion.section>
    </AnimatePresence>
  </main>
}

function Landing({ onStart }: { onStart: () => void }) { return <div className="slide-hero"><div><span className="kicker"><span className="kicker-line" /> SINGLE-PAGE API EXPLORATION</span><h1>See every route your <em>API</em> can take.</h1><p className="hero-lede">Move from contract to topology to findings without losing the thread. This demo keeps every workflow visible while the Python backend remains offline.</p><div className="hero-actions"><button className="button" onClick={onStart}>Start a project <ArrowRight /></button><button className="text-link" onClick={() => window.location.hash = 'projects'}>Explore workspace <GitBranch /></button></div><p className="demo-note"><ShieldCheck /> Demo build: no backend connected.</p></div><div className="slide-card topology-mini"><span className="eyebrow">OPENAPI IN · COVERAGE MAP OUT</span><NetworkSphere /><strong>One continuous workspace</strong><p>Contract, strategy, run, and report.</p></div></div> }

function Auth({ mode, setMode, onContinue }: { mode: 'login' | 'signup'; setMode: (mode: 'login' | 'signup') => void; onContinue: () => void }) { return <div className="center-panel"><span className="eyebrow">{mode === 'login' ? 'WELCOME BACK' : 'CREATE YOUR WORKSPACE'}</span><h2>{mode === 'login' ? 'Return to your API map.' : 'Start mapping your API.'}</h2><p className="muted">Use the demo account flow. No credentials are sent anywhere.</p><div className="auth-tabs"><button className={mode === 'login' ? 'active' : ''} onClick={() => setMode('login')}>Log in</button><button className={mode === 'signup' ? 'active' : ''} onClick={() => setMode('signup')}>Sign up</button></div><label>Email<input type="email" placeholder="you@company.com" /></label><label>Password<input type="password" placeholder="At least 8 characters" /></label><button className="button full" onClick={onContinue}>{mode === 'login' ? 'Continue to projects' : 'Create demo workspace'} <ArrowRight /></button></div> }

function Projects({ onNew, onOpen }: { onNew: () => void; onOpen: () => void }) { return <div><div className="page-heading"><div><span className="eyebrow">WORKSPACE</span><h2>Your API exploration workspaces.</h2></div><button className="button" onClick={onNew}>New project <ArrowRight /></button></div><div className="project-grid"><article className="slide-card project-card"><DemoLabel /><span className="eyebrow">ORDERS API</span><h3>Orders API</h3><p>OpenAPI 3.0 · 42 endpoints</p><div className="stat-row"><strong>68%</strong><span>coverage</span><span>Last run today</span></div><button className="text-link" onClick={onOpen}>Open project <ArrowRight /></button></article><article className="slide-card project-card muted-card"><span className="eyebrow">PAYMENTS API</span><h3>Payments API</h3><p>OpenAPI 3.0 · 18 endpoints</p><div className="stat-row"><strong>—</strong><span>not run</span></div><button className="text-link" onClick={onNew}>Configure project <ArrowRight /></button></article></div></div> }

function Create({ onNext }: { onNext: () => void }) { return <div className="wizard-panel"><span className="eyebrow">NEW PROJECT · 01 / 04</span><h2>Give your API a place to grow.</h2><div className="wizard-steps"><b className="active">01 Details</b><b>02 Contract</b><b>03 Target</b><b>04 Review</b></div><label>Project name<input defaultValue="Orders API" /></label><label>Description<textarea defaultValue="Explore the order lifecycle and contract behavior." /></label><button className="button" onClick={onNext}>Continue to contract <ArrowRight /></button></div> }

function Configure({ onNext }: { onNext: () => void }) { return <div className="wizard-panel"><span className="eyebrow">RUN SETUP · 03 / 04</span><h2>Configure the target and strategy.</h2><div className="config-grid"><div><label>API base URL<input defaultValue="https://api.demo.example/v1" /></label><label>OpenAPI contract<div className="upload-box"><Upload /><strong>orders.openapi.yaml</strong><small>DEMO DATA · file not uploaded</small></div></label></div><div><span className="field-label">Strategy</span><div className="strategy-list"><button className="strategy active"><WandSparkles /><span><strong>Random</strong><small>Broad baseline exploration</small></span><Check /></button><button className="strategy"><GitBranch /><span><strong>PPO</strong><small>Reinforcement learning policy</small></span></button><button className="strategy"><ShieldCheck /><span><strong>PPO + LLM</strong><small>Guided exploration with reasoning</small></span></button></div></div></div><p className="demo-note"><ShieldCheck /> Capability gate: Python runner unavailable, demo run only.</p><button className="button" onClick={onNext}>Start test run <Play /></button></div> }

function Testing({ progress, started, onStart, onResults }: { progress: number; started: boolean; onStart: () => void; onResults: () => void }) { return <div><div className="page-heading"><div><span className="eyebrow">LIVE TESTING · DEMO RUN</span><h2>Watch the API reveal its shape.</h2></div><button className="button" onClick={started ? onResults : onStart}>{started ? 'View results' : 'Start demo run'} <Play /></button></div><div className="testing-layout"><div className="slide-card test-map"><div className="map-toolbar"><span>Dependency map</span><DemoLabel /></div><div className="map-nodes"><span className="node main">POST /orders</span><i /><span className="node">GET /orders/{'{id}'}</span><i /><span className="node finding">GET /customers/{'{id}'}</span><i /><span className="node">POST /payments</span></div></div><div className="slide-card progress-card"><span className="eyebrow">RUN PROGRESS</span><strong>{progress}%</strong><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><p>{started ? 'Exploring declared and discovered paths…' : 'Ready to start. Backend capability is gated.'}</p><div className="finding-line"><span>◆</span><div><strong>1 finding detected</strong><small>404 undeclared · demo evidence</small></div></div></div></div></div> }

function Results({ onBack }: { onBack: () => void }) { return <div><div className="page-heading"><div><span className="eyebrow">RESULTS · ORDERS API</span><h2>Coverage with context.</h2></div><button className="text-link" onClick={onBack}><ArrowLeft /> Back to testing</button></div><div className="results-grid"><div className="slide-card score-card"><span className="eyebrow">COVERAGE</span><strong>68<span>%</span></strong><p>29 of 42 endpoints reached</p></div><div className="slide-card"><span className="eyebrow">FINDINGS</span><h3>1 contract mismatch</h3><p>GET /orders/{'{orderId}'} returned 404, not declared in the spec.</p><span className="finding-badge">◆ Needs review</span></div><div className="slide-card"><span className="eyebrow">STRATEGY COMPARISON</span><div className="bar-row"><span>Random</span><i style={{ width: '68%' }} /></div><div className="bar-row"><span>PPO</span><i style={{ width: '76%' }} /></div><div className="bar-row"><span>PPO + LLM</span><i style={{ width: '84%' }} /></div></div></div><div className="history-strip"><FileCode2 /><span><strong>Run #04</strong> · Demo history · 42 requests · report available</span><button className="text-link">Open report <ArrowRight /></button></div></div> }
