import { useState, useEffect } from 'react'
import { api } from '../api/client'
import type { AssetDetail, AssetRecommendations } from '../types'

export function useAssetDetail(assetId: string | null) {
  const [detail, setDetail] = useState<AssetDetail | null>(null)
  const [recs, setRecs] = useState<AssetRecommendations | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!assetId) return
    setLoading(true)
    setDetail(null)
    setRecs(null)
    Promise.all([
      api.getAssetDetail(assetId),
      api.getAssetRecommendations(assetId),
    ])
      .then(([d, r]) => { setDetail(d); setRecs(r) })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [assetId])

  return { detail, recs, loading }
}
