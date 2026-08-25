-- =============================================================================
-- Razorpay AI Revenue Recovery Agent - Database Schema (PostgreSQL)
-- =============================================================================

-- Table 1: transactions
-- Stores core transaction state, failure details, and recovery status
CREATE TABLE IF NOT EXISTS transactions (
    id VARCHAR(64) PRIMARY KEY,
    razorpay_payment_id VARCHAR(64) NOT NULL,
    merchant_id VARCHAR(64) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'INR',
    status VARCHAR(32) NOT NULL DEFAULT 'FAILED',
    failure_reason TEXT,
    failure_code VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: retry_attempts
-- Tracks retry history, execution timestamps, and scheduled next attempts (Max 2 rule)
CREATE TABLE IF NOT EXISTS retry_attempts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result VARCHAR(32) NOT NULL,
    next_retry_at TIMESTAMP
);

-- Table 3: recovery_actions
-- Records dispatched interventions (smart retries, payment links, chasers) and payloads
CREATE TABLE IF NOT EXISTS recovery_actions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    action_type VARCHAR(64) NOT NULL,
    action_payload TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance optimization
CREATE INDEX IF NOT EXISTS idx_transactions_payment_id ON transactions(razorpay_payment_id);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_retry_attempts_tx ON retry_attempts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_recovery_actions_tx ON recovery_actions(transaction_id);
