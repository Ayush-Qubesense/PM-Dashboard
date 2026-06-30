import { useState, useEffect } from 'react'
import { api } from '../api/client'
import type { FleetKPIs } from '../types'

export function useFleetKPIs() {
  const [data, setData] = useState<FleetKPIs | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getFleetKPIs()
      .then(setData)
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  return { data, loading, error }
}
