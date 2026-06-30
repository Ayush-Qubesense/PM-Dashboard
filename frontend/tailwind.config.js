/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        sidebar: '#1a2332',
        navy: '#0f1929',
      },
    },
  },
  plugins: [],
}
