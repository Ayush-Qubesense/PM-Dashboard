import {
  LayoutDashboard,
  Package,
  Map,
  Briefcase,
  ClipboardList,
  Activity,
  Wrench,
  Box,
  BarChart2,
  Bell,
  Settings,
} from 'lucide-react'

const NAV_ITEMS = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Assets', icon: Package },
  { label: 'Map', icon: Map },
  { label: 'Jobs', icon: Briefcase },
  { label: 'Work Orders', icon: ClipboardList },
  { label: 'AI Health', icon: Activity, active: true, badge: 'New' },
  { label: 'Maintenance', icon: Wrench },
  { label: 'Parts', icon: Box },
  { label: 'Reports', icon: BarChart2 },
  { label: 'Alerts', icon: Bell },
  { label: 'Settings', icon: Settings },
]

export function Sidebar() {
  return (
    <header className="w-full bg-[#1a2332] flex items-center px-4 h-14 flex-shrink-0 gap-2">
      {/* Logo */}
      <div className="mr-4 flex-shrink-0">
        <div className="text-white font-bold text-lg leading-none">
          Ops<span className="text-blue-400">Flo</span>
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex items-center gap-1 flex-1 overflow-x-auto">
        {NAV_ITEMS.map(item => (
          <NavItem key={item.label} {...item} />
        ))}
      </nav>

      {/* Powered by */}
      <div className="flex items-center gap-2 flex-shrink-0 opacity-60 pl-3 border-l border-white/10">
        <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center">
          <Activity size={14} className="text-white" />
        </div>
        <span className="text-[10px] text-slate-400 leading-tight">
          Powered by
          <br />
          OpsFlo AI
        </span>
      </div>
    </header>
  )
}

function NavItem({
  label,
  icon: Icon,
  active,
  badge,
}: {
  label: string
  icon: React.ComponentType<{ size?: number; className?: string }>
  active?: boolean
  badge?: string
}) {
  return (
    <button
      className={`flex items-center gap-2 py-2 px-3 rounded-lg transition-colors relative whitespace-nowrap ${
        active
          ? 'bg-blue-600/20 text-blue-400'
          : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
      }`}
    >
      <Icon size={18} className={active ? 'text-blue-400' : ''} />
      <span className="text-xs font-medium leading-none">{label}</span>
      {badge && (
        <span className="absolute -top-0.5 -right-0.5 text-[8px] bg-blue-500 text-white px-1 rounded font-bold">
          {badge}
        </span>
      )}
    </button>
  )
}