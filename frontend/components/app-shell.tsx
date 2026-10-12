'use client'

import Link from 'next/link'
import { FilePlus2, FolderKanban, History, LogOut, Settings2 } from 'lucide-react'

export function Brand() {
  return <Link href="/" className="brand" aria-label="Rhizome home"><span className="brand-mark"><span /><span /><span /></span><span>Rhizome</span></Link>
}

export function AppShell({ children, title, eyebrow = 'WORKSPACE' }: { children: React.ReactNode; title: string; eyebrow?: string }) {
  return <div className="app-frame">
    <aside className="app-sidebar">
      <Brand />
      <div className="mock-pill"><span /> DEMO MODE · BACKEND OFFLINE</div>
      <nav className="app-nav" aria-label="Workspace navigation">
        <Link href="/projects"><FolderKanban /> Projects</Link>
        <Link href="/projects/new"><FilePlus2 /> New project</Link>
        <Link href="/projects/history"><History /> Run history</Link>
      </nav>
      <div className="sidebar-bottom"><Link href="/projects/settings"><Settings2 /> Settings</Link><Link href="/"><LogOut /> Exit demo</Link></div>
    </aside>
    <div className="app-content"><header className="app-topbar"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1></div><div className="topbar-user"><span className="avatar">AR</span><span>Alex Rivera</span></div></header>{children}</div>
  </div>
}

export function DemoLabel() { return <span className="demo-label">DEMO DATA · NOT CONNECTED</span> }

export function ProgressSteps({ current }: { current: number }) {
  const steps = ['Details', 'Contract', 'Target', 'Review']
  return <div className="wizard-steps">{steps.map((step, index) => <div key={step} className={index <= current ? 'wizard-step active' : 'wizard-step'}><span>{String(index + 1).padStart(2, '0')}</span>{step}</div>)}</div>
}
