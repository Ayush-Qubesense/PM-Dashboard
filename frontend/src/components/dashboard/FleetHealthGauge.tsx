interface Props {
  score: number
  label: string
}

export function FleetHealthGauge({ score, label }: Props) {
  const cx = 80
  const cy = 80
  const r = 60
  const strokeW = 14

  // Arc helpers — semicircle from 180° to 0° (left to right)
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const arcX = (deg: number) => cx + r * Math.cos(toRad(deg))
  const arcY = (deg: number) => cy + r * Math.sin(toRad(deg))

  // Background arc: 180° → 0° (spans 180°)
  const bgStart = { x: arcX(180), y: arcY(180) }
  const bgEnd = { x: arcX(0), y: arcY(0) }
  const bgPath = `M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 0 1 ${bgEnd.x} ${bgEnd.y}`

  // Score arc: proportion of 180°
  const scoreDeg = 180 - (score / 100) * 180
  const scoreEnd = { x: arcX(scoreDeg), y: arcY(scoreDeg) }
  const largeArc = score > 50 ? 1 : 0
  const scorePath = `M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 ${largeArc} 1 ${scoreEnd.x} ${scoreEnd.y}`

  // Needle
  const needleAngle = 180 - (score / 100) * 180
  const needleLen = r - 10
  const needleX = cx + needleLen * Math.cos(toRad(needleAngle))
  const needleY = cy + needleLen * Math.sin(toRad(needleAngle))

  const scoreColor =
    score >= 75 ? '#22c55e'
    : score >= 50 ? '#f59e0b'
    : score >= 25 ? '#f97316'
    : '#ef4444'

  return (
    <div className="flex flex-col items-center">
      <svg width="160" height="100" viewBox="0 0 160 100">
        {/* Gradient definitions */}
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>

        {/* Background track */}
        <path
          d={bgPath}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />

        {/* Colored gradient track (full) */}
        <path
          d={bgPath}
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth={strokeW}
          strokeLinecap="round"
          opacity="0.25"
        />

        {/* Active score arc */}
        <path
          d={scorePath}
          fill="none"
          stroke={scoreColor}
          strokeWidth={strokeW}
          strokeLinecap="round"
        />

        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needleX}
          y2={needleY}
          stroke="#334155"
          strokeWidth={2}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={4} fill="#334155" />

        {/* Score text */}
        <text
          x={cx}
          y={cy - 12}
          textAnchor="middle"
          fontSize="22"
          fontWeight="700"
          fill="#1e293b"
        >
          {score}
        </text>
        <text
          x={cx}
          y={cy + 4}
          textAnchor="middle"
          fontSize="10"
          fill={scoreColor}
          fontWeight="600"
        >
          {label}
        </text>
      </svg>
    </div>
  )
}
