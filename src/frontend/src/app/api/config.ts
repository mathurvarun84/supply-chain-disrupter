/**
 * Single source of truth for the API base URL used by every fetch call.
 *
 * Defaults to "" (relative paths), which relies on the Vite dev server proxy
 * in vite.config.ts to forward /api/* to the FastAPI backend on port 8173.
 * Set VITE_API_BASE_URL only if the backend runs somewhere other than
 * localhost:8173 (e.g. a deployed API).
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
