'use client'
import Link from 'next/link'
import { ArrowRight, LockKeyhole } from 'lucide-react'
import { Brand, DemoLabel } from '@/components/app-shell'

export default function LoginPage() { return <main className="auth-page"><div className="auth-brand"><Brand /></div><section className="auth-card"><div className="auth-icon"><LockKeyhole /></div><span className="eyebrow">WELCOME BACK</span><h1>Return to your API map.</h1><p className="auth-copy">Sign in to continue exploring your projects and test runs.</p><DemoLabel /><form className="form-stack" onSubmit={(event) => event.preventDefault()}><label>Email address<input type="email" placeholder="you@company.com" defaultValue="alex@northstar.dev" /></label><label>Password<input type="password" placeholder="••••••••" defaultValue="password" /></label><button className="button" type="submit">Log in <ArrowRight /></button></form><p className="auth-foot">New to Rhizome? <Link href="/signup">Create an account</Link></p></section></main> }
