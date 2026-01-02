"""
Drone Memory System

Persistent learning database for combat drones.
Stores mistakes, achievements, strategies, and opponent patterns
across training sessions.

This enables drones to:
- Remember what went wrong in past engagements
- Track successful strategies
- Learn opponent patterns
- Build on previous training runs

Usage:
    memory = DroneMemoryDB("combat_memory.db")

    # Record a mistake
    memory.record_mistake(
        drone_id="red_01",
        mistake_type="stall",
        context={"altitude": 200, "speed": 45},
        severity=0.8,
    )

    # Get drone's history
    stats = memory.get_drone_stats("red_01")
    print(f"K/D: {stats.kill_death_ratio}")
"""

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MistakeType(Enum):
    """Types of combat mistakes."""
    STALL = "stall"                    # Speed dropped below stall
    OVERSPEED = "overspeed"            # Exceeded speed limit
    OUT_OF_BOUNDS = "out_of_bounds"    # Left arena
    CRASH = "crash"                    # Hit ground
    FRIENDLY_FIRE = "friendly_fire"   # Shot teammate
    TUNNEL_VISION = "tunnel_vision"   # Ignored threat from behind
    POOR_ENERGY = "poor_energy"       # Bad energy state before engagement
    WASTED_AMMO = "wasted_ammo"       # Fired at out-of-range target
    PREDICTABLE = "predictable"       # Used same maneuver repeatedly
    OVERCOMMIT = "overcommit"         # Chased too long, got ambushed


class AchievementType(Enum):
    """Types of achievements."""
    FIRST_KILL = "first_kill"
    ACE = "ace"                       # 5+ kills in one match
    DOUBLE_ACE = "double_ace"         # 10+ kills
    SURVIVOR = "survivor"             # Completed match without dying
    MARKSMAN = "marksman"             # 50%+ hit rate
    ENERGY_MASTER = "energy_master"   # Won with energy advantage
    MISSILE_ACE = "missile_ace"       # 5+ missile kills
    GUN_MASTER = "gun_master"         # 10+ gun kills
    CLUTCH = "clutch"                 # Won while outnumbered
    STREAK = "streak"                 # 3+ kills without dying


class ManeuverType(Enum):
    """Types of maneuvers to track."""
    BARREL_ROLL = "barrel_roll"
    SPLIT_S = "split_s"
    IMMELMANN = "immelmann"
    BREAK_TURN = "break_turn"
    HIGH_G_TURN = "high_g_turn"
    DIVE = "dive"
    CLIMB = "climb"
    SCISSORS = "scissors"
    YO_YO = "yo_yo"
    ENERGY_TRAP = "energy_trap"


@dataclass
class MistakeRecord:
    """Record of a single mistake."""
    mistake_id: int
    drone_id: str
    session_id: str
    timestamp: float
    mistake_type: str
    context: Dict[str, Any]
    severity: float  # 0-1
    outcome: str  # "death", "damage", "missed_kill", etc.

    @classmethod
    def from_row(cls, row: tuple) -> "MistakeRecord":
        return cls(
            mistake_id=row[0],
            drone_id=row[1],
            session_id=row[2],
            timestamp=row[3],
            mistake_type=row[4],
            context=json.loads(row[5]) if row[5] else {},
            severity=row[6],
            outcome=row[7],
        )


@dataclass
class AchievementRecord:
    """Record of an achievement."""
    achievement_id: int
    drone_id: str
    session_id: str
    timestamp: float
    achievement_type: str
    context: Dict[str, Any]
    match_id: str

    @classmethod
    def from_row(cls, row: tuple) -> "AchievementRecord":
        return cls(
            achievement_id=row[0],
            drone_id=row[1],
            session_id=row[2],
            timestamp=row[3],
            achievement_type=row[4],
            context=json.loads(row[5]) if row[5] else {},
            match_id=row[6],
        )


@dataclass
class ManeuverStats:
    """Statistics for a specific maneuver."""
    maneuver_type: str
    times_used: int = 0
    times_successful: int = 0  # Led to kill or evasion
    times_failed: int = 0      # Led to death or damage
    avg_g_force: float = 0.0
    avg_speed: float = 0.0
    avg_altitude: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.times_successful + self.times_failed
        return self.times_successful / total if total > 0 else 0.5


@dataclass
class OpponentModel:
    """Model of a specific opponent type."""
    opponent_type: str  # "aggressive", "defensive", "balanced"
    encounters: int = 0
    wins_against: int = 0
    losses_against: int = 0
    avg_engagement_range: float = 500.0
    common_maneuvers: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins_against + self.losses_against
        return self.wins_against / total if total > 0 else 0.5


@dataclass
class DroneStats:
    """Aggregate statistics for a drone."""
    drone_id: str
    total_kills: int = 0
    total_deaths: int = 0
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    total_matches: int = 0
    total_wins: int = 0
    total_training_hours: float = 0.0

    # Derived
    kill_death_ratio: float = 1.0
    win_rate: float = 0.5

    # Strategy preferences (learned over time)
    preferred_altitude: float = 500.0
    preferred_speed: float = 200.0
    preferred_engagement_range: float = 300.0
    aggression_score: float = 0.5  # 0=defensive, 1=aggressive

    # Performance by situation
    head_on_win_rate: float = 0.5
    tail_chase_win_rate: float = 0.5
    defensive_survival_rate: float = 0.5

    # Maneuver success rates
    maneuver_stats: Dict[str, ManeuverStats] = field(default_factory=dict)

    # Opponent models
    opponent_models: Dict[str, OpponentModel] = field(default_factory=dict)

    # Recent mistakes (for learning)
    recent_mistakes: List[str] = field(default_factory=list)

    # Achievements earned
    achievements: List[str] = field(default_factory=list)


class DroneMemoryDB:
    """
    SQLite database for persistent drone learning.

    Stores complete history of:
    - Combat outcomes
    - Mistakes and their contexts
    - Achievements
    - Maneuver success rates
    - Opponent patterns
    """

    def __init__(self, db_path: str = "drone_memory.db"):
        """
        Initialize drone memory database.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to drone memory at {self.db_path}")

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Drone stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drone_stats (
                drone_id TEXT PRIMARY KEY,
                total_kills INTEGER DEFAULT 0,
                total_deaths INTEGER DEFAULT 0,
                total_damage_dealt REAL DEFAULT 0,
                total_damage_taken REAL DEFAULT 0,
                total_matches INTEGER DEFAULT 0,
                total_wins INTEGER DEFAULT 0,
                total_training_hours REAL DEFAULT 0,
                preferred_altitude REAL DEFAULT 500,
                preferred_speed REAL DEFAULT 200,
                preferred_engagement_range REAL DEFAULT 300,
                aggression_score REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Mistakes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mistakes (
                mistake_id INTEGER PRIMARY KEY AUTOINCREMENT,
                drone_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                mistake_type TEXT NOT NULL,
                context TEXT,
                severity REAL DEFAULT 0.5,
                outcome TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (drone_id) REFERENCES drone_stats(drone_id)
            )
        ''')

        # Achievements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS achievements (
                achievement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                drone_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                achievement_type TEXT NOT NULL,
                context TEXT,
                match_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (drone_id) REFERENCES drone_stats(drone_id)
            )
        ''')

        # Maneuver stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maneuver_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drone_id TEXT NOT NULL,
                maneuver_type TEXT NOT NULL,
                times_used INTEGER DEFAULT 0,
                times_successful INTEGER DEFAULT 0,
                times_failed INTEGER DEFAULT 0,
                avg_g_force REAL DEFAULT 0,
                avg_speed REAL DEFAULT 0,
                avg_altitude REAL DEFAULT 0,
                UNIQUE(drone_id, maneuver_type),
                FOREIGN KEY (drone_id) REFERENCES drone_stats(drone_id)
            )
        ''')

        # Opponent models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opponent_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drone_id TEXT NOT NULL,
                opponent_type TEXT NOT NULL,
                encounters INTEGER DEFAULT 0,
                wins_against INTEGER DEFAULT 0,
                losses_against INTEGER DEFAULT 0,
                avg_engagement_range REAL DEFAULT 500,
                common_maneuvers TEXT,
                vulnerabilities TEXT,
                UNIQUE(drone_id, opponent_type),
                FOREIGN KEY (drone_id) REFERENCES drone_stats(drone_id)
            )
        ''')

        # Combat events table (detailed history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS combat_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                drone_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                match_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                attacker_id TEXT,
                target_id TEXT,
                weapon TEXT,
                damage REAL DEFAULT 0,
                position_x REAL,
                position_y REAL,
                position_z REAL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (drone_id) REFERENCES drone_stats(drone_id)
            )
        ''')

        # Training sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                drone_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration_hours REAL DEFAULT 0,
                timesteps INTEGER DEFAULT 0,
                episodes INTEGER DEFAULT 0,
                final_elo REAL,
                notes TEXT,
                FOREIGN KEY (drone_id) REFERENCES drone_stats(drone_id)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mistakes_drone ON mistakes(drone_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_achievements_drone ON achievements(drone_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_drone ON combat_events(drone_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_session ON combat_events(session_id)')

        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # =========================================================================
    # DRONE STATS
    # =========================================================================

    def get_or_create_drone(self, drone_id: str) -> DroneStats:
        """Get or create drone stats."""
        cursor = self.conn.cursor()

        cursor.execute('SELECT * FROM drone_stats WHERE drone_id = ?', (drone_id,))
        row = cursor.fetchone()

        if row:
            stats = DroneStats(
                drone_id=row['drone_id'],
                total_kills=row['total_kills'],
                total_deaths=row['total_deaths'],
                total_damage_dealt=row['total_damage_dealt'],
                total_damage_taken=row['total_damage_taken'],
                total_matches=row['total_matches'],
                total_wins=row['total_wins'],
                total_training_hours=row['total_training_hours'],
                preferred_altitude=row['preferred_altitude'],
                preferred_speed=row['preferred_speed'],
                preferred_engagement_range=row['preferred_engagement_range'],
                aggression_score=row['aggression_score'],
            )
        else:
            # Create new
            cursor.execute(
                'INSERT INTO drone_stats (drone_id) VALUES (?)',
                (drone_id,)
            )
            self.conn.commit()
            stats = DroneStats(drone_id=drone_id)

        # Load derived data
        stats.kill_death_ratio = (
            stats.total_kills / max(stats.total_deaths, 1)
        )
        stats.win_rate = (
            stats.total_wins / max(stats.total_matches, 1)
        )

        # Load maneuver stats
        stats.maneuver_stats = self._load_maneuver_stats(drone_id)

        # Load opponent models
        stats.opponent_models = self._load_opponent_models(drone_id)

        # Load achievements
        stats.achievements = self._load_achievements_list(drone_id)

        # Load recent mistakes
        stats.recent_mistakes = self._load_recent_mistakes(drone_id, limit=10)

        return stats

    def update_drone_stats(
        self,
        drone_id: str,
        kills: int = 0,
        deaths: int = 0,
        damage_dealt: float = 0,
        damage_taken: float = 0,
        match_won: bool = False,
        training_hours: float = 0,
    ):
        """Update drone statistics after a match."""
        cursor = self.conn.cursor()

        # Ensure drone exists
        self.get_or_create_drone(drone_id)

        cursor.execute('''
            UPDATE drone_stats SET
                total_kills = total_kills + ?,
                total_deaths = total_deaths + ?,
                total_damage_dealt = total_damage_dealt + ?,
                total_damage_taken = total_damage_taken + ?,
                total_matches = total_matches + 1,
                total_wins = total_wins + ?,
                total_training_hours = total_training_hours + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE drone_id = ?
        ''', (
            kills, deaths, damage_dealt, damage_taken,
            1 if match_won else 0, training_hours, drone_id
        ))

        self.conn.commit()

    def update_preferences(
        self,
        drone_id: str,
        preferred_altitude: Optional[float] = None,
        preferred_speed: Optional[float] = None,
        preferred_engagement_range: Optional[float] = None,
        aggression_score: Optional[float] = None,
    ):
        """Update learned preferences."""
        cursor = self.conn.cursor()

        updates = []
        values = []

        if preferred_altitude is not None:
            updates.append("preferred_altitude = ?")
            values.append(preferred_altitude)
        if preferred_speed is not None:
            updates.append("preferred_speed = ?")
            values.append(preferred_speed)
        if preferred_engagement_range is not None:
            updates.append("preferred_engagement_range = ?")
            values.append(preferred_engagement_range)
        if aggression_score is not None:
            updates.append("aggression_score = ?")
            values.append(max(0, min(1, aggression_score)))

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            values.append(drone_id)

            cursor.execute(
                f"UPDATE drone_stats SET {', '.join(updates)} WHERE drone_id = ?",
                tuple(values)
            )
            self.conn.commit()

    # =========================================================================
    # MISTAKES
    # =========================================================================

    def record_mistake(
        self,
        drone_id: str,
        mistake_type: str,
        context: Dict[str, Any],
        severity: float = 0.5,
        outcome: str = "damage",
        session_id: str = "",
    ) -> int:
        """
        Record a mistake for learning.

        Args:
            drone_id: Drone identifier
            mistake_type: Type of mistake (see MistakeType)
            context: Contextual information (position, speed, enemy positions)
            severity: Severity 0-1 (1 = fatal)
            outcome: What happened ("death", "damage", "missed_kill")
            session_id: Training session ID

        Returns:
            Mistake ID
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO mistakes (drone_id, session_id, timestamp, mistake_type, context, severity, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            drone_id,
            session_id or f"session_{int(time.time())}",
            time.time(),
            mistake_type,
            json.dumps(context),
            severity,
            outcome,
        ))

        self.conn.commit()
        mistake_id = cursor.lastrowid

        logger.debug(f"Recorded mistake {mistake_type} for {drone_id}")
        return mistake_id

    def get_mistakes(
        self,
        drone_id: str,
        mistake_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[MistakeRecord]:
        """Get mistake history."""
        cursor = self.conn.cursor()

        if mistake_type:
            cursor.execute('''
                SELECT * FROM mistakes
                WHERE drone_id = ? AND mistake_type = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (drone_id, mistake_type, limit))
        else:
            cursor.execute('''
                SELECT * FROM mistakes
                WHERE drone_id = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (drone_id, limit))

        return [MistakeRecord.from_row(tuple(row)) for row in cursor.fetchall()]

    def _load_recent_mistakes(self, drone_id: str, limit: int = 10) -> List[str]:
        """Load recent mistake types."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT mistake_type FROM mistakes
            WHERE drone_id = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (drone_id, limit))
        return [row[0] for row in cursor.fetchall()]

    # =========================================================================
    # ACHIEVEMENTS
    # =========================================================================

    def record_achievement(
        self,
        drone_id: str,
        achievement_type: str,
        context: Dict[str, Any],
        match_id: str = "",
        session_id: str = "",
    ) -> int:
        """
        Record an achievement.

        Args:
            drone_id: Drone identifier
            achievement_type: Type of achievement (see AchievementType)
            context: Contextual information
            match_id: Match where achieved
            session_id: Training session ID

        Returns:
            Achievement ID
        """
        cursor = self.conn.cursor()

        # Check if already earned
        cursor.execute('''
            SELECT 1 FROM achievements
            WHERE drone_id = ? AND achievement_type = ?
        ''', (drone_id, achievement_type))

        if cursor.fetchone():
            return -1  # Already has this achievement

        cursor.execute('''
            INSERT INTO achievements (drone_id, session_id, timestamp, achievement_type, context, match_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            drone_id,
            session_id or f"session_{int(time.time())}",
            time.time(),
            achievement_type,
            json.dumps(context),
            match_id,
        ))

        self.conn.commit()
        achievement_id = cursor.lastrowid

        logger.info(f"Achievement unlocked: {achievement_type} for {drone_id}")
        return achievement_id

    def _load_achievements_list(self, drone_id: str) -> List[str]:
        """Load list of achievement types earned."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT achievement_type FROM achievements
            WHERE drone_id = ?
        ''', (drone_id,))
        return [row[0] for row in cursor.fetchall()]

    # =========================================================================
    # MANEUVER STATS
    # =========================================================================

    def record_maneuver(
        self,
        drone_id: str,
        maneuver_type: str,
        successful: bool,
        g_force: float = 0,
        speed: float = 0,
        altitude: float = 0,
    ):
        """Record maneuver usage."""
        cursor = self.conn.cursor()

        # Upsert
        cursor.execute('''
            INSERT INTO maneuver_stats (drone_id, maneuver_type, times_used, times_successful, times_failed, avg_g_force, avg_speed, avg_altitude)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(drone_id, maneuver_type) DO UPDATE SET
                times_used = times_used + 1,
                times_successful = times_successful + ?,
                times_failed = times_failed + ?,
                avg_g_force = (avg_g_force * times_used + ?) / (times_used + 1),
                avg_speed = (avg_speed * times_used + ?) / (times_used + 1),
                avg_altitude = (avg_altitude * times_used + ?) / (times_used + 1)
        ''', (
            drone_id, maneuver_type,
            1 if successful else 0, 0 if successful else 1,
            g_force, speed, altitude,
            1 if successful else 0, 0 if successful else 1,
            g_force, speed, altitude,
        ))

        self.conn.commit()

    def _load_maneuver_stats(self, drone_id: str) -> Dict[str, ManeuverStats]:
        """Load maneuver statistics."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM maneuver_stats WHERE drone_id = ?', (drone_id,))

        stats = {}
        for row in cursor.fetchall():
            ms = ManeuverStats(
                maneuver_type=row['maneuver_type'],
                times_used=row['times_used'],
                times_successful=row['times_successful'],
                times_failed=row['times_failed'],
                avg_g_force=row['avg_g_force'],
                avg_speed=row['avg_speed'],
                avg_altitude=row['avg_altitude'],
            )
            stats[row['maneuver_type']] = ms

        return stats

    # =========================================================================
    # OPPONENT MODELS
    # =========================================================================

    def update_opponent_model(
        self,
        drone_id: str,
        opponent_type: str,
        won: bool,
        engagement_range: float = 0,
        opponent_maneuvers: Optional[List[str]] = None,
        vulnerabilities: Optional[List[str]] = None,
    ):
        """Update model of an opponent type."""
        cursor = self.conn.cursor()

        # Check existing
        cursor.execute('''
            SELECT * FROM opponent_models
            WHERE drone_id = ? AND opponent_type = ?
        ''', (drone_id, opponent_type))

        row = cursor.fetchone()

        if row:
            # Update existing
            encounters = row['encounters'] + 1
            wins = row['wins_against'] + (1 if won else 0)
            losses = row['losses_against'] + (0 if won else 1)

            # Running average for engagement range
            new_range = (row['avg_engagement_range'] * row['encounters'] + engagement_range) / encounters

            # Merge maneuvers
            existing_maneuvers = json.loads(row['common_maneuvers'] or '[]')
            if opponent_maneuvers:
                for m in opponent_maneuvers:
                    if m not in existing_maneuvers:
                        existing_maneuvers.append(m)

            # Merge vulnerabilities
            existing_vulns = json.loads(row['vulnerabilities'] or '[]')
            if vulnerabilities:
                for v in vulnerabilities:
                    if v not in existing_vulns:
                        existing_vulns.append(v)

            cursor.execute('''
                UPDATE opponent_models SET
                    encounters = ?,
                    wins_against = ?,
                    losses_against = ?,
                    avg_engagement_range = ?,
                    common_maneuvers = ?,
                    vulnerabilities = ?
                WHERE drone_id = ? AND opponent_type = ?
            ''', (
                encounters, wins, losses, new_range,
                json.dumps(existing_maneuvers[-10:]),  # Keep last 10
                json.dumps(existing_vulns[-5:]),
                drone_id, opponent_type,
            ))
        else:
            # Create new
            cursor.execute('''
                INSERT INTO opponent_models (drone_id, opponent_type, encounters, wins_against, losses_against, avg_engagement_range, common_maneuvers, vulnerabilities)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ''', (
                drone_id, opponent_type,
                1 if won else 0, 0 if won else 1,
                engagement_range,
                json.dumps(opponent_maneuvers or []),
                json.dumps(vulnerabilities or []),
            ))

        self.conn.commit()

    def _load_opponent_models(self, drone_id: str) -> Dict[str, OpponentModel]:
        """Load opponent models."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM opponent_models WHERE drone_id = ?', (drone_id,))

        models = {}
        for row in cursor.fetchall():
            om = OpponentModel(
                opponent_type=row['opponent_type'],
                encounters=row['encounters'],
                wins_against=row['wins_against'],
                losses_against=row['losses_against'],
                avg_engagement_range=row['avg_engagement_range'],
                common_maneuvers=json.loads(row['common_maneuvers'] or '[]'),
                vulnerabilities=json.loads(row['vulnerabilities'] or '[]'),
            )
            models[row['opponent_type']] = om

        return models

    # =========================================================================
    # COMBAT EVENTS
    # =========================================================================

    def record_combat_event(
        self,
        drone_id: str,
        event_type: str,
        session_id: str,
        match_id: str,
        attacker_id: Optional[str] = None,
        target_id: Optional[str] = None,
        weapon: Optional[str] = None,
        damage: float = 0,
        position: Optional[Tuple[float, float, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Record a combat event."""
        cursor = self.conn.cursor()

        pos_x, pos_y, pos_z = position if position else (0, 0, 0)

        cursor.execute('''
            INSERT INTO combat_events (drone_id, session_id, match_id, timestamp, event_type, attacker_id, target_id, weapon, damage, position_x, position_y, position_z, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            drone_id, session_id, match_id, time.time(),
            event_type, attacker_id, target_id, weapon, damage,
            pos_x, pos_y, pos_z,
            json.dumps(context) if context else None,
        ))

        self.conn.commit()

    # =========================================================================
    # TRAINING SESSIONS
    # =========================================================================

    def start_session(self, drone_id: str, session_id: Optional[str] = None) -> str:
        """Start a training session."""
        if session_id is None:
            session_id = f"session_{int(time.time())}_{drone_id[:8]}"

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO training_sessions (session_id, drone_id)
            VALUES (?, ?)
        ''', (session_id, drone_id))

        self.conn.commit()
        logger.info(f"Started training session {session_id}")
        return session_id

    def end_session(
        self,
        session_id: str,
        timesteps: int = 0,
        episodes: int = 0,
        final_elo: Optional[float] = None,
        notes: str = "",
    ):
        """End a training session."""
        cursor = self.conn.cursor()

        # Calculate duration
        cursor.execute('SELECT started_at FROM training_sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        if row:
            started = datetime.fromisoformat(row[0])
            duration = (datetime.now() - started).total_seconds() / 3600

            cursor.execute('''
                UPDATE training_sessions SET
                    ended_at = CURRENT_TIMESTAMP,
                    duration_hours = ?,
                    timesteps = ?,
                    episodes = ?,
                    final_elo = ?,
                    notes = ?
                WHERE session_id = ?
            ''', (duration, timesteps, episodes, final_elo, notes, session_id))

            self.conn.commit()
            logger.info(f"Ended session {session_id} ({duration:.2f} hours)")

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def get_weakness_report(self, drone_id: str) -> Dict[str, Any]:
        """Analyze drone's weaknesses based on mistake history."""
        cursor = self.conn.cursor()

        # Count mistakes by type
        cursor.execute('''
            SELECT mistake_type, COUNT(*) as count, AVG(severity) as avg_severity
            FROM mistakes WHERE drone_id = ?
            GROUP BY mistake_type
            ORDER BY count DESC
        ''', (drone_id,))

        weaknesses = []
        for row in cursor.fetchall():
            weaknesses.append({
                'type': row['mistake_type'],
                'count': row['count'],
                'avg_severity': row['avg_severity'],
            })

        # Get worst maneuvers
        cursor.execute('''
            SELECT maneuver_type, times_used, times_failed,
                   CAST(times_failed AS REAL) / NULLIF(times_used, 0) as fail_rate
            FROM maneuver_stats WHERE drone_id = ?
            ORDER BY fail_rate DESC
        ''', (drone_id,))

        weak_maneuvers = []
        for row in cursor.fetchall():
            if row['times_used'] >= 5:  # Minimum sample size
                weak_maneuvers.append({
                    'maneuver': row['maneuver_type'],
                    'attempts': row['times_used'],
                    'failures': row['times_failed'],
                    'fail_rate': row['fail_rate'] or 0,
                })

        # Toughest opponents
        cursor.execute('''
            SELECT opponent_type, encounters, wins_against, losses_against,
                   CAST(losses_against AS REAL) / NULLIF(encounters, 0) as loss_rate
            FROM opponent_models WHERE drone_id = ?
            ORDER BY loss_rate DESC
        ''', (drone_id,))

        tough_opponents = []
        for row in cursor.fetchall():
            if row['encounters'] >= 3:
                tough_opponents.append({
                    'opponent': row['opponent_type'],
                    'encounters': row['encounters'],
                    'loss_rate': row['loss_rate'] or 0,
                })

        return {
            'drone_id': drone_id,
            'top_weaknesses': weaknesses[:5],
            'weak_maneuvers': weak_maneuvers[:3],
            'tough_opponents': tough_opponents[:3],
            'recommendations': self._generate_recommendations(weaknesses, weak_maneuvers),
        }

    def _generate_recommendations(
        self,
        weaknesses: List[Dict],
        weak_maneuvers: List[Dict],
    ) -> List[str]:
        """Generate training recommendations."""
        recommendations = []

        for w in weaknesses[:3]:
            mistake = w['type']
            if mistake == 'stall':
                recommendations.append("Focus on energy management - maintain speed above stall")
            elif mistake == 'tunnel_vision':
                recommendations.append("Improve situational awareness - check six o'clock regularly")
            elif mistake == 'predictable':
                recommendations.append("Vary tactics - use different approach angles")
            elif mistake == 'overcommit':
                recommendations.append("Practice disengagement - know when to break off")
            elif mistake == 'poor_energy':
                recommendations.append("Enter engagements with energy advantage")

        for m in weak_maneuvers[:2]:
            maneuver = m['maneuver']
            recommendations.append(f"Practice {maneuver} - currently {m['fail_rate']*100:.0f}% fail rate")

        return recommendations

    def export_drone_profile(self, drone_id: str) -> Dict[str, Any]:
        """Export complete drone profile for analysis."""
        stats = self.get_or_create_drone(drone_id)
        weaknesses = self.get_weakness_report(drone_id)

        return {
            'drone_id': drone_id,
            'stats': asdict(stats),
            'weaknesses': weaknesses,
            'exported_at': datetime.now().isoformat(),
        }
