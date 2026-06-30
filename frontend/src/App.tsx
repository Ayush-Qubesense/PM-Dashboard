import { useState } from 'react'
import { Sidebar } from './components/layout/Sidebar'
import { Header } from './components/layout/Header'
import { FleetKPICards } from './components/dashboard/FleetKPICards'
import { AssetHealthList } from './components/dashboard/AssetHealthList'
import { AssetDetailPanel } from './components/dashboard/AssetDetailPanel'
import { AIRecommendations } from './components/dashboard/AIRecommendations'
import { PMJobOptimization } from './components/dashboard/PMJobOptimization'
import { useFleetKPIs } from './hooks/useFleetKPIs'
import { useAssetDetail } from './hooks/useAssetDetail'

export default function App() {
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null)
  const [dateRange, setDateRange] = useState('7d')

  const { data: kpis, loading: kpisLoading } = useFleetKPIs()
  const { detail, recs, loading: detailLoading } = useAssetDetail(selectedAssetId)

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#f1f5f9]">
      <Sidebar />
      <Header dateRange={dateRange} onDateRangeChange={setDateRange} />

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <main className="flex-1 overflow-auto p-4">
          {/* KPI cards */}
          {kpisLoading || !kpis ? (
            <div className="h-[140px] flex items-center justify-center text-slate-400 text-sm mb-4">
              Loading fleet metrics...
            </div>
          ) : (
            <FleetKPICards kpis={kpis} />
          )}

          {/* Three-panel content area */}
          <div className="grid grid-cols-[380px_1fr_300px] gap-4 h-[calc(100vh-300px)] min-h-[500px]">
            {/* Left: Asset Health List */}
            <AssetHealthList
              selectedId={selectedAssetId}
              onSelect={setSelectedAssetId}
            />

            {/* Middle: Asset Detail */}
            <AssetDetailPanel detail={detail} loading={detailLoading} />

            {/* Right: Recommendations + PM */}
            <div className="flex flex-col gap-4">
              <AIRecommendations recs={recs} loading={detailLoading} />
              <PMJobOptimization pmJob={recs?.pm_job ?? null} loading={detailLoading} />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
