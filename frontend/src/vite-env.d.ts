/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  // Day 5: the frontend's own auth-mode contract. Must be kept
  // consistent with the backend's AUTH_MODE (see .env.example). Unset is
  // treated as "disabled" so existing local development is unchanged.
  readonly VITE_AUTH_MODE?: string
  // Firebase public web config -- required only when VITE_AUTH_MODE is
  // "firebase". These are public client-side values (they ship in the
  // browser bundle by design); they are NOT secrets. No service-account
  // key or private value belongs here.
  readonly VITE_FIREBASE_API_KEY?: string
  readonly VITE_FIREBASE_AUTH_DOMAIN?: string
  readonly VITE_FIREBASE_PROJECT_ID?: string
  readonly VITE_FIREBASE_APP_ID?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
