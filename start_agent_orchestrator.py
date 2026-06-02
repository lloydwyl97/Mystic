#!/usr/bin/env python3
"""
Start the Agent Orchestrator with Social Media Agent
This runs the actual agent orchestrator that includes social media monitoring
"""

# ============================================================================
# CRITICAL FIX: Process Lock - Prevent Duplicate Orchestrators on Restart
# ============================================================================
import fcntl
import os
import signal
import time
from pathlib import Path

LOCK_FILE = "/tmp/mystic_orchestrator.lock"


def cleanup_lock_file():
    """Clean up lock file on exit"""
    try:
        lock_path = Path(LOCK_FILE)
        if lock_path.exists():
            lock_path.unlink()
    except Exception as e:
        print(f"Warning: Could not remove lock file: {e}")


def ensure_single_instance():
    """
    Ensure only one orchestrator is running.
    If another instance exists, kill it gracefully before starting new one.
    """
    try:
        # Try to open/create lock file
        lock = open(LOCK_FILE, "w")

        # Try to acquire exclusive lock (non-blocking)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Successfully acquired lock - we're the only instance
            lock.write(str(os.getpid()))
            lock.flush()
            print(f"ORCHESTRATOR_SINGLETON_ACQUIRED pid={os.getpid()} lock_file={LOCK_FILE}")
            print(f"✅ Acquired orchestrator lock (PID: {os.getpid()})")
            return lock
        except OSError:
            # Lock is held by another process
            print("⚠️ Another orchestrator instance detected. Terminating old instance...")

            # Read PID from lock file
            try:
                lock.seek(0)
                old_pid_str = lock.read().strip()
                if old_pid_str:
                    old_pid = int(old_pid_str)
                    print(f"   Old PID: {old_pid}")

                    # Check if process exists
                    try:
                        os.kill(old_pid, 0)  # Check if process exists
                        print(f"   Sending SIGTERM to PID {old_pid}...")
                        os.kill(old_pid, signal.SIGTERM)
                        time.sleep(2)  # Give it time to shutdown

                        # If still alive, SIGKILL
                        try:
                            os.kill(old_pid, 0)
                            print("   Process still alive, sending SIGKILL...")
                            os.kill(old_pid, signal.SIGKILL)
                            time.sleep(1)
                        except ProcessLookupError:
                            print("   Process terminated successfully")
                    except ProcessLookupError:
                        print(f"   Process {old_pid} not found (already dead)")
            except Exception as e:
                print(f"   Error reading old PID: {e}")

            # Now acquire lock for new instance
            lock.truncate(0)
            lock.seek(0)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            lock.write(str(os.getpid()))
            lock.flush()
            print(f"ORCHESTRATOR_SINGLETON_ACQUIRED pid={os.getpid()} lock_file={LOCK_FILE}")
            print(f"✅ Acquired orchestrator lock for new instance (PID: {os.getpid()})")
            return lock

    except Exception as e:
        print(f"❌ ERROR acquiring lock: {e}")
        print("   Continuing anyway (singleton protection disabled)")
        return None


# ============================================================================
# CRITICAL FIX: Windows ProactorEventLoop has bugs with async Redis connections
# Must be set BEFORE any asyncio operations
# ============================================================================
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import asyncio

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# ACQUIRE LOCK BEFORE ANYTHING ELSE
lock_file = ensure_single_instance()


async def main():
    """Start the agent orchestrator"""
    try:
        print("=" * 60)
        print("STARTING AGENT ORCHESTRATOR (Social Media + NLP Agents)")
        print("=" * 60)

        # Import the agent orchestrator
        from backend.agents.agent_orchestrator import AgentOrchestrator

        print("\nInitializing Agent Orchestrator...")
        orchestrator = AgentOrchestrator()

        print("Starting orchestrator and all agents...")
        await orchestrator.start()

        print("\n✅ AGENT ORCHESTRATOR STARTED!")
        print("   - Social Media Agent: ACTIVE")
        print("   - News Sentiment Agent: ACTIVE")
        print("   - Market Sentiment Agent: ACTIVE")
        print("   - Strategy, Risk, Execution, Compliance Agents: ACTIVE")
        print("\nSocial sentiment will be published to Redis every 3 minutes.")
        print("Press Ctrl+C to stop.\n")

        # Keep running
        while True:
            await asyncio.sleep(10)

            # Show agent status periodically
            if hasattr(orchestrator, "state") and "agent_status" in orchestrator.state:
                status = orchestrator.state["agent_status"]
                running_agents = [name for name, state in status.items() if state == "running"]
                if running_agents:
                    print(f"✓ Active agents: {len(running_agents)} - {', '.join(running_agents[:3])}...")

    except KeyboardInterrupt:
        print("\n\nShutting down Agent Orchestrator...")
        if orchestrator:
            await orchestrator.stop()
        print("Stopped.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always clean up lock file on exit
        cleanup_lock_file()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cleanup_lock_file()
