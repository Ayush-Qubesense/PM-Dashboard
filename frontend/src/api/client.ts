import type {
  FleetKPIs,
  AssetHealthList,
  AssetDetail,
  AssetRecommendations,
  SortOption,
  RiskLevel,
} from '../types'

// Local dev: VITE_API_BASE_URL is unset → Vite proxies /api to localhost:8000
// Vercel prod: set VITE_API_BASE_URL=https://your-app.onrender.com/api
const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json() as Promise<T>
}

export async function post(path: string): Promise<unknown> {
  const res = await fetch(BASE + path, { method: 'POST' })
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json()
}

export const api = {
  getFleetKPIs: () => get<FleetKPIs>('/fleet/kpis'),

  getAssetHealthList: (params: {
    sort?: SortOption
    risk_level?: RiskLevel | null
    page?: number
    limit?: number
  }) => {
    const q = new URLSearchParams()
    if (params.sort) q.set('sort', params.sort)
    if (params.risk_level) q.set('risk_level', params.risk_level)
    if (params.page) q.set('page', String(params.page))
    if (params.limit) q.set('limit', String(params.limit))
    return get<AssetHealthList>(`/fleet/assets/health?${q}`)
  },

  getAssetDetail: (assetId: string) =>
    get<AssetDetail>(`/assets/${assetId}/detail`),

  getAssetRecommendations: (assetId: string) =>
    get<AssetRecommendations>(`/assets/${assetId}/recommendations`),

  acceptRecommendation: (recId: string) =>
    post(`/recommendations/${recId}/accept`),

  dismissRecommendation: (recId: string) =>
    post(`/recommendations/${recId}/dismiss`),
}
