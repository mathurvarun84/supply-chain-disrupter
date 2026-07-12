/**
 * Typed fetch functions for the Admin page — GET/POST /api/admin/*.
 * Backed by src/api/routers/admin.py.
 */
import type { AdminJobTriggerResponse, AdminStatusResponse } from "../types/admin";
import { API_BASE_URL } from "./config";

export const fetchAdminStatus = async (): Promise<AdminStatusResponse> => {
  const res = await fetch(`${API_BASE_URL}/api/admin/status`);
  if (!res.ok) throw new Error(`admin/status failed: ${res.status}`);
  return res.json();
};

export const postDatabaseBuild = async (): Promise<AdminJobTriggerResponse> => {
  const res = await fetch(`${API_BASE_URL}/api/admin/db/build`, { method: "POST" });
  if (!res.ok) throw new Error(`admin/db/build failed: ${res.status}`);
  return res.json();
};

export const postRagBuild = async (flush: boolean): Promise<AdminJobTriggerResponse> => {
  const res = await fetch(`${API_BASE_URL}/api/admin/rag/build?flush=${flush}`, { method: "POST" });
  if (!res.ok) throw new Error(`admin/rag/build failed: ${res.status}`);
  return res.json();
};
