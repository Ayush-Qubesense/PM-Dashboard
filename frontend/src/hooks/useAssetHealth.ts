import { useState, useEffect } from 'react'
import { api } from '../api/client'
import type { AssetHealthList, SortOption, RiskLevel } from '../types'

export function useAssetHealth(
  sort: SortOption,
  riskFilter: RiskLevel | null,
  page: number,
) {
  const [data, setData] = useState<AssetHealthList | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api.getAssetHealthList({ sort, risk_level: riskFilter, page, limit: 5 })
      .then(setData)
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [sort, riskFilter, page])

  return { data, loading, error }
}
