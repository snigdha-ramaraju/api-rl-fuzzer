'use client'

import { useEffect, useState } from 'react'

const GLOBE_IMAGE = 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-JhlMbgEGae5HBKfzbxBqd2OJRUJp1L.png'

export function NetworkSphere() {
  const [paused, setPaused] = useState(false)
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false)

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setPrefersReducedMotion(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  const isPaused = paused || prefersReducedMotion

  return (
    <div className={`network-sphere ${isPaused ? 'is-paused' : ''}`} aria-label="Emerald globe made of interconnected API particles">
      <div className="network-glow" aria-hidden="true" />
      <img className="network-globe-image" src={GLOBE_IMAGE} alt="Translucent emerald globe showing interconnected routes across the Americas" />
      <div className="network-caption" aria-hidden="true"><span>CONNECTED SURFACE</span><i /> <span>API TOPOLOGY</span></div>
      <button className="network-motion-toggle" type="button" onClick={() => setPaused((value) => !value)} aria-pressed={isPaused}>{isPaused ? 'Motion off' : 'Pause motion'}</button>
    </div>
  )
}
