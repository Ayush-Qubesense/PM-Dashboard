import { Bell, HelpCircle, ChevronDown, Cpu } from 'lucide-react'

interface HeaderProps {
  dateRange: string
  onDateRangeChange: (v: string) => void
}

export function Header({ dateRange, onDateRangeChange }: HeaderProps) {
  return (
    <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between flex-shrink-0">
      {/* Left: title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-slate-800">
          <Cpu size={18} className="text-blue-600" />
          <span className="font-semibold text-base">AI Health &amp; Predictive Maintenance</span>
        </div>
        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-semibold">Beta</span>
      </div>

      {/* Right: actions */}
      <div className="flex items-center gap-4">
        {/* Date range */}
        <select
          value={dateRange}
          onChange={e => onDateRangeChange(e.target.value)}
          className="text-sm border border-slate-200 rounded px-3 py-1.5 text-slate-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
          <option value="90d">Last 90 Days</option>
        </select>

        <button className="text-slate-500 hover:text-slate-700 relative">
          <Bell size={18} />
          <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" />
        </button>
        <button className="text-slate-500 hover:text-slate-700">
          <HelpCircle size={18} />
        </button>

        {/* Fleet selector */}
        <button className="flex items-center gap-2 border border-slate-200 rounded px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
          <span className="font-medium">NG Fleet</span>
          <ChevronDown size={14} />
        </button>
      </div>
    </header>
  )
}
