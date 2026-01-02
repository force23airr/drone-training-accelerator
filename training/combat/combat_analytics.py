"""
Combat Analytics and Strategy Extraction

Analyzes dogfight data to extract winning strategies and combat patterns.
Provides insights for drone companies to understand optimal fighting tactics.

Features:
- Engagement pattern detection (head-on, tail chase, ambush, etc.)
- Maneuver classification (split-S, barrel roll, Immelmann, etc.)
- Kill/death analysis by situation
- Strategy clustering and naming
- Tactical recommendations
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import logging

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


logger = logging.getLogger(__name__)


# =============================================================================
# ENGAGEMENT PATTERNS
# =============================================================================

class EngagementType:
    """Types of aerial engagements."""
    HEAD_ON = "head_on"           # Flying directly at each other
    TAIL_CHASE = "tail_chase"     # Behind enemy, pursuing
    AMBUSH = "ambush"             # Surprise attack from advantageous position
    SCISSORS = "scissors"          # Rolling scissors dogfight
    VERTICAL = "vertical"          # Vertical maneuvering fight
    MERGE = "merge"               # Initial merge after head-on pass
    DEFENSIVE = "defensive"        # Being chased, evading
    NEUTRAL = "neutral"           # Neither has advantage


class ManeuverType:
    """Common air combat maneuvers."""
    BARREL_ROLL = "barrel_roll"
    SPLIT_S = "split_s"
    IMMELMANN = "immelmann"
    BREAK_TURN = "break_turn"
    HIGH_YO_YO = "high_yo_yo"
    LOW_YO_YO = "low_yo_yo"
    LAG_PURSUIT = "lag_pursuit"
    LEAD_PURSUIT = "lead_pursuit"
    PURE_PURSUIT = "pure_pursuit"
    ENERGY_TRAP = "energy_trap"
    VERTICAL_EXTENSION = "vertical_extension"
    DIVE = "dive"
    CLIMB = "climb"
    LEVEL_TURN = "level_turn"


@dataclass
class EngagementRecord:
    """Record of a single engagement."""
    timestamp: float
    engagement_type: str
    attacker_id: int
    defender_id: int
    relative_position: np.ndarray  # Relative position of defender to attacker
    relative_velocity: np.ndarray  # Relative velocity
    attacker_altitude: float
    defender_altitude: float
    attacker_speed: float
    defender_speed: float
    angle_off_tail: float  # 0 = directly behind, 180 = head-on
    aspect_angle: float    # Angle from defender's perspective
    range: float           # Distance between aircraft
    result: str           # "hit", "miss", "evade"
    weapon_used: str
    damage_dealt: float = 0.0


@dataclass
class ManeuverRecord:
    """Record of a detected maneuver."""
    timestamp: float
    maneuver_type: str
    drone_id: int
    duration: float
    altitude_change: float
    speed_change: float
    g_force_peak: float
    energy_state_before: float  # Total energy (KE + PE)
    energy_state_after: float
    success: bool  # Whether it achieved tactical goal


@dataclass
class StrategyProfile:
    """Profile of a discovered strategy."""
    strategy_id: str
    name: str
    description: str
    engagement_preferences: Dict[str, float]  # Preferred engagement types
    maneuver_frequencies: Dict[str, float]    # How often each maneuver is used
    altitude_preference: str  # "high", "medium", "low"
    speed_preference: str     # "fast", "medium", "slow"
    aggression_score: float   # 0-1, how aggressive
    patience_score: float     # 0-1, how patient
    energy_management: float  # 0-1, how well energy is managed
    kill_rate: float
    survival_rate: float
    avg_engagement_range: float
    effectiveness_score: float


@dataclass
class CombatAnalytics:
    """Aggregate combat analytics."""
    total_engagements: int = 0
    total_kills: int = 0
    total_deaths: int = 0
    engagement_breakdown: Dict[str, int] = field(default_factory=dict)
    maneuver_breakdown: Dict[str, int] = field(default_factory=dict)
    kill_by_engagement: Dict[str, int] = field(default_factory=dict)
    kill_by_maneuver: Dict[str, int] = field(default_factory=dict)
    avg_kill_range: float = 0.0
    avg_kill_angle: float = 0.0
    optimal_strategies: List[StrategyProfile] = field(default_factory=list)


class CombatAnalyzer:
    """
    Analyzes dogfight data to extract strategies and patterns.
    """

    def __init__(self, data_dir: str):
        """
        Initialize combat analyzer.

        Args:
            data_dir: Directory containing combat logs
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.engagements: List[EngagementRecord] = []
        self.maneuvers: List[ManeuverRecord] = []
        self.analytics = CombatAnalytics()

        self._load_data()

    def _load_data(self):
        """Load existing analytics data."""
        analytics_file = self.data_dir / "combat_analytics.json"
        if analytics_file.exists():
            with open(analytics_file, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.analytics, key):
                        setattr(self.analytics, key, value)

    def save_data(self):
        """Save analytics data."""
        analytics_file = self.data_dir / "combat_analytics.json"
        with open(analytics_file, "w") as f:
            json.dump(asdict(self.analytics), f, indent=2, default=str)

    def classify_engagement(
        self,
        attacker_pos: np.ndarray,
        attacker_vel: np.ndarray,
        defender_pos: np.ndarray,
        defender_vel: np.ndarray,
    ) -> str:
        """
        Classify the type of engagement based on geometry.

        Args:
            attacker_pos: Attacker position
            attacker_vel: Attacker velocity
            defender_pos: Defender position
            defender_vel: Defender velocity

        Returns:
            Engagement type
        """
        # Calculate relative geometry
        to_defender = defender_pos - attacker_pos
        distance = np.linalg.norm(to_defender)

        if distance < 1e-6:
            return EngagementType.NEUTRAL

        to_defender_norm = to_defender / distance

        # Attacker heading
        attacker_speed = np.linalg.norm(attacker_vel)
        if attacker_speed < 1e-6:
            attacker_heading = np.array([1, 0, 0])
        else:
            attacker_heading = attacker_vel / attacker_speed

        # Defender heading
        defender_speed = np.linalg.norm(defender_vel)
        if defender_speed < 1e-6:
            defender_heading = np.array([1, 0, 0])
        else:
            defender_heading = defender_vel / defender_speed

        # Angle off tail (from attacker's perspective)
        # 0 = behind, 180 = head-on
        angle_to_defender = np.arccos(np.clip(
            np.dot(attacker_heading, to_defender_norm), -1, 1
        ))
        angle_off_tail = np.degrees(angle_to_defender)

        # Aspect angle (from defender's perspective)
        aspect = np.arccos(np.clip(
            np.dot(defender_heading, -to_defender_norm), -1, 1
        ))
        aspect_angle = np.degrees(aspect)

        # Closure rate
        closing = -np.dot(attacker_vel - defender_vel, to_defender_norm)

        # Classify based on angles:
        # angle_off_tail: 0 = pointing at defender, 180 = pointing away
        # aspect_angle: 0 = defender facing attacker, 180 = defender facing away

        if angle_off_tail < 30 and aspect_angle < 30:
            # Both pointing at each other - head-on
            return EngagementType.HEAD_ON

        elif angle_off_tail < 30 and aspect_angle > 150:
            # Attacker pointing at defender, defender facing away - tail chase
            return EngagementType.TAIL_CHASE

        elif angle_off_tail > 150 and aspect_angle < 30:
            # Attacker facing away, defender pointing at attacker - defensive
            return EngagementType.DEFENSIVE

        elif 60 < angle_off_tail < 120 and 60 < aspect_angle < 120:
            # Perpendicular, maneuvering
            if abs(attacker_pos[2] - defender_pos[2]) > 100:
                return EngagementType.VERTICAL
            else:
                return EngagementType.SCISSORS

        elif closing > attacker_speed * 0.5:
            return EngagementType.MERGE

        else:
            return EngagementType.NEUTRAL

    def detect_maneuver(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        dt: float = 1/60,
    ) -> Optional[str]:
        """
        Detect maneuver from trajectory data.

        Args:
            positions: Array of positions over time (N x 3)
            velocities: Array of velocities over time (N x 3)
            dt: Time step

        Returns:
            Detected maneuver type or None
        """
        if len(positions) < 10:
            return None

        # Calculate derived quantities
        speeds = np.linalg.norm(velocities, axis=1)
        altitudes = positions[:, 2]

        # Altitude changes
        alt_change = altitudes[-1] - altitudes[0]
        max_alt = np.max(altitudes)
        min_alt = np.min(altitudes)
        alt_range = max_alt - min_alt

        # Speed changes
        speed_change = speeds[-1] - speeds[0]
        avg_speed = np.mean(speeds)

        # Heading changes
        headings = velocities / (speeds[:, np.newaxis] + 1e-6)
        heading_changes = np.arccos(np.clip(
            np.sum(headings[:-1] * headings[1:], axis=1), -1, 1
        ))
        total_turn = np.degrees(np.sum(heading_changes))

        # G-force estimation
        accelerations = np.diff(velocities, axis=0) / dt
        g_forces = np.linalg.norm(accelerations, axis=1) / 9.81
        max_g = np.max(g_forces) if len(g_forces) > 0 else 0

        # Classify based on characteristics
        if alt_change > 200 and speed_change < -20:
            # Nose up, losing speed
            return ManeuverType.IMMELMANN

        elif alt_change < -200 and speed_change > 20:
            # Nose down, gaining speed
            return ManeuverType.SPLIT_S

        elif total_turn > 270 and alt_range > 100:
            # Full roll with altitude change
            return ManeuverType.BARREL_ROLL

        elif max_g > 5 and total_turn > 90 and abs(alt_change) < 50:
            # High-G level turn
            return ManeuverType.BREAK_TURN

        elif alt_change > 100 and total_turn > 45:
            # Climb with turn
            return ManeuverType.HIGH_YO_YO

        elif alt_change < -100 and total_turn > 45:
            # Dive with turn
            return ManeuverType.LOW_YO_YO

        elif alt_change > 300:
            return ManeuverType.VERTICAL_EXTENSION

        elif alt_change < -200:
            return ManeuverType.DIVE

        elif alt_change > 100:
            return ManeuverType.CLIMB

        elif total_turn > 90:
            return ManeuverType.LEVEL_TURN

        return None

    def analyze_trajectory(
        self,
        drone_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        enemy_positions: np.ndarray,
        enemy_velocities: np.ndarray,
        kills: List[Dict[str, Any]],
        deaths: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze a full trajectory and extract patterns.

        Args:
            drone_id: ID of the drone being analyzed
            positions: Drone positions (N x 3)
            velocities: Drone velocities (N x 3)
            enemy_positions: Enemy positions (N x M x 3) for M enemies
            enemy_velocities: Enemy velocities (N x M x 3)
            kills: List of kill events
            deaths: List of death events

        Returns:
            Analysis results
        """
        n_timesteps = len(positions)

        analysis = {
            "drone_id": drone_id,
            "duration": n_timesteps / 60,  # Assuming 60 Hz
            "kills": len(kills),
            "deaths": len(deaths),
            "engagements": defaultdict(int),
            "maneuvers": defaultdict(int),
            "avg_altitude": np.mean(positions[:, 2]),
            "avg_speed": np.mean(np.linalg.norm(velocities, axis=1)),
            "altitude_range": np.ptp(positions[:, 2]),
            "distance_traveled": 0,
            "energy_efficiency": 0,
        }

        # Calculate distance traveled
        if n_timesteps > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            analysis["distance_traveled"] = np.sum(distances)

        # Classify engagements at each timestep
        for t in range(n_timesteps):
            if enemy_positions.ndim == 3:
                for e in range(enemy_positions.shape[1]):
                    eng_type = self.classify_engagement(
                        positions[t],
                        velocities[t],
                        enemy_positions[t, e],
                        enemy_velocities[t, e],
                    )
                    analysis["engagements"][eng_type] += 1

        # Detect maneuvers in sliding windows
        window_size = 60  # 1 second at 60 Hz
        for start in range(0, n_timesteps - window_size, window_size // 2):
            end = start + window_size
            maneuver = self.detect_maneuver(
                positions[start:end],
                velocities[start:end],
            )
            if maneuver:
                analysis["maneuvers"][maneuver] += 1

        return analysis

    def extract_strategies(
        self,
        match_analyses: List[Dict[str, Any]],
        n_clusters: int = 5,
    ) -> List[StrategyProfile]:
        """
        Extract strategy profiles from multiple match analyses using clustering.

        Args:
            match_analyses: List of match analysis results
            n_clusters: Number of strategy clusters to find

        Returns:
            List of discovered strategy profiles
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, using simple strategy extraction")
            return self._simple_strategy_extraction(match_analyses)

        if len(match_analyses) < n_clusters:
            return self._simple_strategy_extraction(match_analyses)

        # Create feature vectors
        features = []
        for analysis in match_analyses:
            feature = [
                analysis.get("avg_altitude", 500) / 1000,
                analysis.get("avg_speed", 100) / 200,
                analysis.get("kills", 0),
                analysis.get("deaths", 0),
                len(analysis.get("engagements", {})),
                len(analysis.get("maneuvers", {})),
                analysis.get("distance_traveled", 0) / 10000,
            ]

            # Add engagement type frequencies
            eng = analysis.get("engagements", {})
            total_eng = sum(eng.values()) or 1
            for eng_type in [EngagementType.HEAD_ON, EngagementType.TAIL_CHASE,
                            EngagementType.DEFENSIVE, EngagementType.VERTICAL]:
                feature.append(eng.get(eng_type, 0) / total_eng)

            features.append(feature)

        features = np.array(features)

        # Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        # Create strategy profiles for each cluster
        strategies = []
        strategy_names = [
            "Aggressive Pursuer",
            "Energy Fighter",
            "Defensive Ace",
            "Balanced Tactician",
            "Hit and Run",
        ]

        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_analyses = [a for a, m in zip(match_analyses, cluster_mask) if m]

            if not cluster_analyses:
                continue

            # Aggregate cluster stats
            avg_kills = np.mean([a.get("kills", 0) for a in cluster_analyses])
            avg_deaths = np.mean([a.get("deaths", 0) for a in cluster_analyses])
            avg_altitude = np.mean([a.get("avg_altitude", 500) for a in cluster_analyses])
            avg_speed = np.mean([a.get("avg_speed", 100) for a in cluster_analyses])

            # Engagement preferences
            eng_counts = defaultdict(int)
            for a in cluster_analyses:
                for eng_type, count in a.get("engagements", {}).items():
                    eng_counts[eng_type] += count
            total = sum(eng_counts.values()) or 1
            eng_prefs = {k: v / total for k, v in eng_counts.items()}

            # Maneuver frequencies
            man_counts = defaultdict(int)
            for a in cluster_analyses:
                for man_type, count in a.get("maneuvers", {}).items():
                    man_counts[man_type] += count
            total = sum(man_counts.values()) or 1
            man_freqs = {k: v / total for k, v in man_counts.items()}

            # Determine characteristics
            if avg_altitude > 1000:
                alt_pref = "high"
            elif avg_altitude < 500:
                alt_pref = "low"
            else:
                alt_pref = "medium"

            if avg_speed > 200:
                speed_pref = "fast"
            elif avg_speed < 100:
                speed_pref = "slow"
            else:
                speed_pref = "medium"

            # Scores
            aggression = eng_prefs.get(EngagementType.HEAD_ON, 0) + \
                        eng_prefs.get(EngagementType.TAIL_CHASE, 0)
            patience = 1 - aggression

            kill_rate = avg_kills / max(avg_kills + avg_deaths, 1)
            survival_rate = 1 - (avg_deaths / max(avg_kills + avg_deaths, 1))

            effectiveness = (kill_rate + survival_rate) / 2

            strategy = StrategyProfile(
                strategy_id=f"strategy_{i}",
                name=strategy_names[i % len(strategy_names)],
                description=self._generate_strategy_description(
                    eng_prefs, man_freqs, alt_pref, speed_pref, aggression
                ),
                engagement_preferences=dict(eng_prefs),
                maneuver_frequencies=dict(man_freqs),
                altitude_preference=alt_pref,
                speed_preference=speed_pref,
                aggression_score=aggression,
                patience_score=patience,
                energy_management=0.5,  # Would need more data to calculate
                kill_rate=kill_rate,
                survival_rate=survival_rate,
                avg_engagement_range=500,  # Would need engagement data
                effectiveness_score=effectiveness,
            )

            strategies.append(strategy)

        # Sort by effectiveness
        strategies.sort(key=lambda s: s.effectiveness_score, reverse=True)

        return strategies

    def _simple_strategy_extraction(
        self,
        match_analyses: List[Dict[str, Any]],
    ) -> List[StrategyProfile]:
        """Simple strategy extraction without sklearn."""
        if not match_analyses:
            return []

        # Just create one aggregate strategy
        avg_kills = np.mean([a.get("kills", 0) for a in match_analyses])
        avg_deaths = np.mean([a.get("deaths", 0) for a in match_analyses])

        return [StrategyProfile(
            strategy_id="aggregate",
            name="General Strategy",
            description="Aggregate strategy from all matches",
            engagement_preferences={},
            maneuver_frequencies={},
            altitude_preference="medium",
            speed_preference="medium",
            aggression_score=0.5,
            patience_score=0.5,
            energy_management=0.5,
            kill_rate=avg_kills / max(avg_kills + avg_deaths, 1),
            survival_rate=1 - avg_deaths / max(avg_kills + avg_deaths, 1),
            avg_engagement_range=500,
            effectiveness_score=0.5,
        )]

    def _generate_strategy_description(
        self,
        eng_prefs: Dict[str, float],
        man_freqs: Dict[str, float],
        alt_pref: str,
        speed_pref: str,
        aggression: float,
    ) -> str:
        """Generate human-readable strategy description."""
        parts = []

        # Aggression level
        if aggression > 0.7:
            parts.append("Highly aggressive fighter")
        elif aggression > 0.4:
            parts.append("Balanced aggression")
        else:
            parts.append("Defensive and patient")

        # Altitude
        if alt_pref == "high":
            parts.append("prefers high altitude for energy advantage")
        elif alt_pref == "low":
            parts.append("stays low to use terrain")
        else:
            parts.append("maintains medium altitude")

        # Speed
        if speed_pref == "fast":
            parts.append("maintains high speed for boom and zoom attacks")
        elif speed_pref == "slow":
            parts.append("trades speed for maneuverability")

        # Top engagements
        if eng_prefs:
            top_eng = max(eng_prefs.items(), key=lambda x: x[1])[0]
            if top_eng == EngagementType.HEAD_ON:
                parts.append("favors head-on attacks")
            elif top_eng == EngagementType.TAIL_CHASE:
                parts.append("excels at tail chases")
            elif top_eng == EngagementType.VERTICAL:
                parts.append("uses vertical maneuvering")

        return ". ".join(parts) + "."

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        report = {
            "summary": {
                "total_engagements": self.analytics.total_engagements,
                "total_kills": self.analytics.total_kills,
                "total_deaths": self.analytics.total_deaths,
                "kill_death_ratio": (
                    self.analytics.total_kills / max(self.analytics.total_deaths, 1)
                ),
            },
            "engagement_analysis": self.analytics.engagement_breakdown,
            "maneuver_analysis": self.analytics.maneuver_breakdown,
            "kill_analysis": {
                "by_engagement": self.analytics.kill_by_engagement,
                "by_maneuver": self.analytics.kill_by_maneuver,
                "avg_range": self.analytics.avg_kill_range,
                "avg_angle": self.analytics.avg_kill_angle,
            },
            "strategies": [
                asdict(s) for s in self.analytics.optimal_strategies
            ],
            "recommendations": self._generate_recommendations(),
            "generated_at": datetime.now().isoformat(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate tactical recommendations based on analytics."""
        recommendations = []

        if not self.analytics.optimal_strategies:
            return ["Insufficient data for recommendations"]

        best_strategy = self.analytics.optimal_strategies[0]

        recommendations.append(
            f"Most effective strategy: {best_strategy.name} "
            f"(effectiveness: {best_strategy.effectiveness_score:.1%})"
        )

        if best_strategy.aggression_score > 0.6:
            recommendations.append(
                "Aggressive tactics are working - maintain pressure on opponents"
            )
        else:
            recommendations.append(
                "Defensive tactics showing success - prioritize survival"
            )

        if best_strategy.altitude_preference == "high":
            recommendations.append(
                "High altitude provides energy advantage - maintain altitude"
            )
        elif best_strategy.altitude_preference == "low":
            recommendations.append(
                "Low altitude tactics working - continue terrain masking"
            )

        # Engagement recommendations
        eng_prefs = best_strategy.engagement_preferences
        if eng_prefs.get(EngagementType.TAIL_CHASE, 0) > 0.3:
            recommendations.append(
                "Tail chase engagements are highly effective - seek rear aspect"
            )
        if eng_prefs.get(EngagementType.HEAD_ON, 0) > 0.3:
            recommendations.append(
                "Head-on attacks working well - continue offensive merges"
            )

        return recommendations


def analyze_training_run(training_dir: str) -> Dict[str, Any]:
    """
    Analyze a completed training run.

    Args:
        training_dir: Directory containing training outputs

    Returns:
        Analysis report
    """
    analyzer = CombatAnalyzer(training_dir)
    return analyzer.generate_report()
