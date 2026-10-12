'use client'
import Link from 'next/link'
import { ArrowRight, UserRoundPlus } from 'lucide-react'
import { Brand, DemoLabel } from '@/components/app-shell'

export default function SignupPage() { return <main className="auth-page"><div className="auth-brand"><Brand /></div><section className="auth-card"><div className="auth-icon"><UserRoundPlus /></div><span className="eyebrow">START EXPLORING</span><h1>Build a clearer API map.</h1><p className="auth-copy">Create a workspace for specs, targets, and test runs.</p><DemoLabel /><form className="form-stack" onSubmit={(event) => event.preventDefault()}><label>Full name<input placeholder="Alex Rivera" /></label><label>Work email<input type="email" placeholder="you@company.com" /></label><label>Password<input type="password" placeholder="At least 8 characters" /></label><button className="button" type="submit">Create account <ArrowRight /></button></form><p className="auth-foot">Already have an account? <Link href="/login">Log in</Link></p></section></main> }
