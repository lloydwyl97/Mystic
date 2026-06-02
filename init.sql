-- Mystic Trading Platform Database Initialization
-- This script sets up the initial database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create database user with limited privileges (for application use)
-- Note: This should be run by a superuser, then the application should use the mystic_app user
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mystic_app') THEN
      CREATE ROLE mystic_app LOGIN PASSWORD 'mystic_secure_2024';
   END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE mystic_trading TO mystic_app;
GRANT USAGE ON SCHEMA public TO mystic_app;

-- Create audit log table (for PostgreSQL-based audit storage)
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    resource VARCHAR(255),
    action VARCHAR(100),
    status VARCHAR(50),
    details JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_category ON audit_logs (category);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_severity ON audit_logs (severity);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs (resource);

-- Create partition table for better performance with large audit logs
-- This will automatically partition by month
CREATE TABLE IF NOT EXISTS audit_logs_y2024m01 PARTITION OF audit_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    response_time FLOAT NOT NULL,
    status_code INTEGER NOT NULL,
    request_size INTEGER,
    response_size INTEGER,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for performance metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_endpoint ON performance_metrics (endpoint);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_status_code ON performance_metrics (status_code);

-- Create trading signals table (for analytics)
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
    confidence FLOAT NOT NULL,
    price FLOAT,
    volume BIGINT,
    indicators JSONB,
    metadata JSONB
);

-- Create indexes for trading signals
CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals (symbol);
CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_type ON trading_signals (signal_type);

-- Create user sessions table (backup to Redis)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    data JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Create indexes for sessions
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions (session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions (expires_at);

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mystic_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mystic_app;

-- Create views for analytics
CREATE OR REPLACE VIEW daily_api_usage AS
SELECT
    DATE(timestamp) as date,
    endpoint,
    method,
    COUNT(*) as total_requests,
    AVG(response_time) as avg_response_time,
    MIN(response_time) as min_response_time,
    MAX(response_time) as max_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
FROM performance_metrics
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(timestamp), endpoint, method
ORDER BY date DESC, total_requests DESC;

CREATE OR REPLACE VIEW trading_signal_performance AS
SELECT
    DATE(timestamp) as date,
    symbol,
    signal_type,
    COUNT(*) as signal_count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM trading_signals
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(timestamp), symbol, signal_type
ORDER BY date DESC, signal_count DESC;

-- Create maintenance functions
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(days_old INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM audit_logs
    WHERE timestamp < NOW() - INTERVAL '1 day' * days_old;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions
    WHERE expires_at < NOW() OR is_active = FALSE;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic cleanup (optional, can be run via cron)
-- Note: In production, these should be run via a separate maintenance job

-- Insert sample data for testing (remove in production)
INSERT INTO audit_logs (event_id, event_type, category, severity, resource, action, status, details)
VALUES
    ('test-001', 'system_startup', 'system', 'info', 'application', 'start', 'success',
     '{"message": "Application started successfully", "version": "1.0.0"}')
ON CONFLICT DO NOTHING;

-- Log successful database initialization
DO $$
BEGIN
    INSERT INTO audit_logs (event_id, event_type, category, severity, resource, action, status, details)
    VALUES (
        'db-init-' || uuid_generate_v4()::text,
        'database_initialization',
        'system',
        'info',
        'database',
        'initialize',
        'success',
        jsonb_build_object(
            'message', 'Database schema initialized successfully',
            'tables_created', ARRAY['audit_logs', 'performance_metrics', 'trading_signals', 'user_sessions'],
            'indexes_created', 10,
            'views_created', ARRAY['daily_api_usage', 'trading_signal_performance']
        )
    );
END $$;
