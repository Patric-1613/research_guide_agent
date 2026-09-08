import CurationWorkspacePage from './pages/CurationWorkspacePage'
import { AuthProvider } from './lib/auth/AuthProvider'
import { AuthGate } from './components/Auth/AuthGate'

export default function App() {
  return (
    <AuthProvider>
      <AuthGate>
        <CurationWorkspacePage />
      </AuthGate>
    </AuthProvider>
  )
}
