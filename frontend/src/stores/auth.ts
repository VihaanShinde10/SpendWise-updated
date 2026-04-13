import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi } from '../lib/api';

interface User {
  id: string;
  email: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      login: async (email, password) => {
        const res = await authApi.login(email, password);
        const { access_token, user_id } = res.data;
        localStorage.setItem('sb-access-token', access_token);
        localStorage.setItem('sb-user-id', user_id);
        set({
          user: { id: user_id, email },
          token: access_token,
          isAuthenticated: true,
        });
      },

      register: async (email, password, name) => {
        const res = await authApi.register(email, password, name);
        // After registration, auto-login
        const loginRes = await authApi.login(email, password);
        const { access_token, user_id } = loginRes.data;
        localStorage.setItem('sb-access-token', access_token);
        localStorage.setItem('sb-user-id', user_id);
        set({
          user: { id: user_id, email },
          token: access_token,
          isAuthenticated: true,
        });
      },

      logout: () => {
        localStorage.removeItem('sb-access-token');
        localStorage.removeItem('sb-user-id');
        set({ user: null, token: null, isAuthenticated: false });
      },
    }),
    { name: 'spendwise-auth' }
  )
);
