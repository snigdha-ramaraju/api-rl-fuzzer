import Link from 'next/link'
import { SlidingApp } from '@/components/sliding-app'
import { NetworkSphere } from '@/components/network-sphere'
import { ArrowRight, Check, ChevronRight, CircleHelp, FileCode2, GitBranch, Menu, Play, Search, Sparkles } from 'lucide-react'

const nodes = [
  { x: 54, y: 68, label: 'createOrder', tone: 'main' },
  { x: 184, y: 38, label: 'getOrder', tone: 'soft' },
  { x: 292, y: 102, label: 'cancelOrder', tone: 'soft' },
  { x: 160, y: 160, label: 'listOrders', tone: 'soft' },
  { x: 316, y: 214, label: 'getCustomer', tone: 'soft' },
  { x: 90, y: 245, label: 'createPayment', tone: 'main' },
  { x: 242, y: 294, label: 'getPayment', tone: 'soft' },
]

const edges = [[54, 68, 184, 38], [184, 38, 292, 102], [54, 68, 160, 160], [160, 160, 90, 245], [90, 245, 242, 294], [292, 102, 316, 214], [160, 160, 316, 214]]

function TopologyPreview() {
  return (
    <div className="topology-card" aria-label="Illustrated API topology preview">
      <div className="topology-toolbar">
        <span className="eyebrow">OPENAPI IN · COVERAGE MAP OUT</span>
        <span className="toolbar-dot"><span /> LIVE STRUCTURE</span>
      </div>
      <svg className="topology-svg" viewBox="0 0 370 350" role="img" aria-label="A network of API endpoints connected by dependency edges">
        <defs>
          <filter id="softGlow"><feGaussianBlur stdDeviation="7" /></filter>
          <linearGradient id="edgeFade" x1="0" x2="1"><stop stopColor="#bfe3d4" /><stop offset="1" stopColor="#087f5b" /></linearGradient>
        </defs>
        <circle cx="185" cy="170" r="104" fill="#e5f5ef" opacity=".7" filter="url(#softGlow)" />
        {edges.map(([x1, y1, x2, y2], index) => <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} stroke="url(#edgeFade)" strokeWidth="1.5" strokeDasharray={index === 1 ? '4 5' : undefined} />)}
        <path d="M54 68 C98 102 124 112 160 160" fill="none" stroke="#087f5b" strokeWidth="2" strokeDasharray="3 8" opacity=".65" />
        {nodes.map((node, index) => <g key={node.label} className="topology-node">
          <circle cx={node.x} cy={node.y} r={node.tone === 'main' ? 15 : 11} fill={node.tone === 'main' ? '#087f5b' : '#fff'} stroke={node.tone === 'main' ? '#056044' : '#2e9b78'} strokeWidth="2" />
          {node.tone === 'main' && <circle cx={node.x} cy={node.y} r="20" fill="none" stroke="#bfe3d4" strokeDasharray="2 3" />}
          <text x={node.x + 19} y={node.y + 4} className="node-label">{node.label}</text>
          {index === 1 && <path d={`M${node.x} ${node.y - 19} l6 6 -6 6 -6-6 z`} fill="#c58a24" />}
        </g>)}
        <g transform="translate(238 118)"><path d="M0 0 l7 7 -7 7 -7-7 z" fill="#c58a24" /><text x="15" y="5" className="finding-label">404 undeclared</text></g>
      </svg>
      <div className="topology-caption"><span><i className="legend-dot covered" /> reached</span><span><i className="legend-dot finding" /> finding</span><span><i className="legend-line" /> dependency</span></div>
    </div>
  )
}

function Step({ number, icon: Icon, title, children }: { number: string; icon: typeof FileCode2; title: string; children: string }) {
  return <article className="step-card"><div className="step-number">{number}</div><Icon className="step-icon" strokeWidth={1.5} /><h3>{title}</h3><p>{children}</p><span className="step-arrow"><ArrowRight /></span></article>
}

function LegacyLanding() {
  return (
    <main>
      <header className="site-header">
        <Link href="/" className="brand" aria-label="Rhizome home"><span className="brand-mark"><span /><span /><span /></span><span>Rhizome</span></Link>
        <nav className="desktop-nav" aria-label="Main navigation"><Link href="#how-it-works">How it works</Link><Link href="#sample">Sample report</Link><Link href="#docs">Docs</Link></nav>
        <div className="header-actions"><Link href="/login" className="login-link">Log in</Link><Link href="/projects/new" className="button button-small">Start a project <ArrowRight /></Link></div>
        <button className="mobile-menu" aria-label="Open menu"><Menu /></button>
      </header>

      <section className="hero section-shell">
        <div className="hero-copy"><div className="kicker"><span className="kicker-line" /> SPEC-CONFORMANCE, MADE VISIBLE</div><h1>See every route your <em>API</em> can take.</h1><p className="hero-lede">Rhizome explores your API the way a careful user would, maps what has been covered, and flags where the API disagrees with its own contract.</p><div className="hero-actions"><Link href="/projects/new" className="button">Start a project <ArrowRight /></Link><Link href="#sample" className="text-link"><Play /> Explore a sample run</Link></div><p className="demo-note"><Sparkles /> Demo build: no backend connected.</p></div>
        <div className="hero-visual"><NetworkSphere /><div className="hero-stamp"><span>01</span><span>MAP THE CONTRACT</span></div></div>
      </section>

      <section className="proof-strip section-shell"><div><Check /> Reads OpenAPI 3.x</div><div><Check /> Coverage per endpoint and status</div><div><Check /> Contract mismatches, with the request</div></section>

      <section id="how-it-works" className="how section-shell"><div className="section-heading"><span className="eyebrow">A CLEARER WAY IN</span><h2>From contract to <em>clarity.</em></h2><p>Rhizome turns a static OpenAPI file into a living map of your API. Explore the shape, then understand the gaps.</p></div><div className="steps"><Step number="01" icon={FileCode2} title="Upload" >Your spec becomes a map. Endpoints appear as nodes. Dependencies draw themselves as lines.</Step><Step number="02" icon={GitBranch} title="Explore">Coverage spreads along real routes. Watch each node fill as its declared responses are seen.</Step><Step number="03" icon={Search} title="Read">Findings sit on the structure. Unexplored gaps stay visible, not hidden in a table.</Step></div></section>

      <section id="sample" className="sample section-shell"><div className="sample-card"><div className="sample-copy"><span className="eyebrow">SAMPLE REPORT · ORDERS API</span><h2>Not just a log.<br /><em>A living topology.</em></h2><p>See how a run moves through your contract. Every edge is a dependency, every gold mark is a place where the response and the spec diverge.</p><Link href="/projects/sample" className="text-link">View sample report <ChevronRight /></Link></div><div className="finding-plate"><div className="plate-header"><span>PLATE II</span><span>FINDING · HTTP 404</span></div><div className="plate-path">GET <strong>/orders/{'{orderId}'}</strong></div><div className="plate-rule" /><div className="plate-detail"><span className="diamond">◆</span><div><strong>Status not in the spec</strong><p>The API answered 404, but the spec does not list 404 for this endpoint.</p></div></div><div className="plate-request">GET /orders/order_00000 <span>→ 404 · 118 ms</span></div></div></div></section>

      <footer className="site-footer section-shell"><Link href="/" className="brand"><span className="brand-mark"><span /><span /><span /></span><span>Rhizome</span></Link><p>Understand the shape of your API.</p><div className="footer-links"><Link href="#how-it-works">How it works</Link><Link href="#sample">Sample report</Link><Link href="/login">Log in</Link></div>      </footer>
    </main>
  )
}

export default function Page() {
  return <SlidingApp />
}

