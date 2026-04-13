-- ==========================================
-- SpendWise Full Database Setup Script
-- Paste this into your Supabase SQL Editor
-- ==========================================

-- 1. Enable Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Users Profile Table
CREATE TABLE IF NOT EXISTS public.profiles (
  id            UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email         TEXT UNIQUE NOT NULL,
  display_name  TEXT,
  currency      TEXT DEFAULT 'INR',
  timezone      TEXT DEFAULT 'Asia/Kolkata',
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_profile" ON public.profiles
  FOR ALL USING (auth.uid() = id);

-- 3. Categories Table
CREATE TABLE IF NOT EXISTS public.categories (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id     UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  icon        TEXT,
  color       TEXT,
  is_system   BOOLEAN DEFAULT FALSE,
  parent_id   UUID REFERENCES public.categories(id),
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.categories ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_categories" ON public.categories
  FOR ALL USING (auth.uid() = user_id OR is_system = TRUE);

INSERT INTO public.categories (name, icon, color, is_system, user_id) VALUES
  ('Food & Dining',        '🍽️',  '#FF6B6B', TRUE, NULL),
  ('Transport',            '🚗',  '#4ECDC4', TRUE, NULL),
  ('Shopping',             '🛍️',  '#45B7D1', TRUE, NULL),
  ('Entertainment',        '🎬',  '#96CEB4', TRUE, NULL),
  ('Utilities',            '⚡',  '#FFEAA7', TRUE, NULL),
  ('Health & Medical',     '🏥',  '#DDA0DD', TRUE, NULL),
  ('Education',            '📚',  '#98D8C8', TRUE, NULL),
  ('Travel',               '✈️',  '#F7DC6F', TRUE, NULL),
  ('Financial Services',   '💳',  '#82E0AA', TRUE, NULL),
  ('Groceries',            '🛒',  '#F8C471', TRUE, NULL),
  ('Peer Transfer',        '👤',  '#AED6F1', TRUE, NULL),
  ('Others',               '📦',  '#D5DBDB', TRUE, NULL)
ON CONFLICT DO NOTHING;

-- 4. Transactions Table
CREATE TABLE IF NOT EXISTS public.transactions (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id               UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  raw_description       TEXT NOT NULL,
  amount                NUMERIC(14, 2) NOT NULL,
  direction             TEXT CHECK (direction IN ('debit', 'credit')) NOT NULL,
  balance               NUMERIC(14, 2),
  transaction_date      TIMESTAMPTZ NOT NULL,
  payment_method        TEXT CHECK (payment_method IN ('UPI', 'IMPS', 'NEFT', 'ATM', 'OTHER')),
  cleaned_description   TEXT,
  merchant_name         TEXT,
  is_low_descriptiveness BOOLEAN DEFAULT FALSE,
  category_id           UUID REFERENCES public.categories(id),
  category_source       TEXT CHECK (category_source IN ('semantic','behavioural','fused','zero_shot','manual','pending')),
  confidence_score      NUMERIC(4, 3),
  gating_alpha          NUMERIC(4, 3),
  is_recurring          BOOLEAN DEFAULT FALSE,
  recurrence_strength   NUMERIC(4, 3),
  needs_review          BOOLEAN DEFAULT FALSE,
  user_corrected        BOOLEAN DEFAULT FALSE,
  user_category_id      UUID REFERENCES public.categories(id),
  processing_status     TEXT DEFAULT 'pending' CHECK (
    processing_status IN ('pending', 'processing', 'completed', 'failed')
  ),
  processed_at          TIMESTAMPTZ,
  created_at            TIMESTAMPTZ DEFAULT NOW(),
  updated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_transactions_user_date ON public.transactions(user_id, transaction_date DESC);
CREATE INDEX idx_transactions_category ON public.transactions(user_id, category_id);
CREATE INDEX idx_transactions_status ON public.transactions(user_id, processing_status);
CREATE INDEX idx_transactions_merchant ON public.transactions(user_id, merchant_name);

ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_transactions" ON public.transactions
  FOR ALL USING (auth.uid() = user_id);

-- 5. Embeddings Table
CREATE TABLE IF NOT EXISTS public.transaction_embeddings (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  transaction_id  UUID UNIQUE NOT NULL REFERENCES public.transactions(id) ON DELETE CASCADE,
  user_id         UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  embedding       vector(768),
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON public.transaction_embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

ALTER TABLE public.transaction_embeddings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_embeddings" ON public.transaction_embeddings
  FOR ALL USING (auth.uid() = user_id);

-- 6. Cluster State Table
CREATE TABLE IF NOT EXISTS public.user_clusters (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  cluster_id      INTEGER NOT NULL,
  category_id     UUID REFERENCES public.categories(id),
  label_name      TEXT,
  transaction_count INTEGER DEFAULT 0,
  last_clustered  TIMESTAMPTZ DEFAULT NOW(),
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, cluster_id)
);

ALTER TABLE public.user_clusters ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_clusters" ON public.user_clusters
  FOR ALL USING (auth.uid() = user_id);

-- 7. Budgets Table
CREATE TABLE IF NOT EXISTS public.budgets (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id       UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  category_id   UUID NOT NULL REFERENCES public.categories(id),
  amount        NUMERIC(14, 2) NOT NULL,
  period        TEXT CHECK (period IN ('monthly', 'weekly', 'annual')) DEFAULT 'monthly',
  start_date    DATE NOT NULL,
  end_date      DATE,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.budgets ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_budgets" ON public.budgets
  FOR ALL USING (auth.uid() = user_id);

-- 8. Gating Model Weights Table
CREATE TABLE IF NOT EXISTS public.gating_model_state (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID UNIQUE NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  W1              JSONB,
  b1              JSONB,
  W2              JSONB,
  b2              FLOAT,
  training_samples INTEGER DEFAULT 0,
  val_mse         FLOAT,
  trained_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.gating_model_state ENABLE ROW LEVEL SECURITY;
CREATE POLICY "users_own_model" ON public.gating_model_state
  FOR ALL USING (auth.uid() = user_id);

-- 9. Triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_transactions_updated_at
  BEFORE UPDATE ON public.transactions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_budgets_updated_at
  BEFORE UPDATE ON public.budgets
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Note: Run this last manually or ensure publication exists
-- ALTER PUBLICATION supabase_realtime ADD TABLE public.transactions;
