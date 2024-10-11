import gym
# import pybullet_envs
# from .isaacgym_envs.envs.shadow_hand_reach import ShadowHandReachEnv
# from .isaacgym_envs.envs.shadow_hand_finger_reach import ShadowHandFingerReachEnv
# from .isaacgym_envs.envs.shadow_hand_reorient import ShadowHandReorientEnv

class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        # make myosuite envs
        env_map = {
            "MyoFingerPoseFixed":               "myoFingerPoseFixed-v0",
            "MyoFingerPoseRandom":              "myoFingerPoseRandom-v0",
            "MyoFingerReachFixed":              "myoFingerReachFixed-v0",
            "MyoFingerReachRandom":             "myoFingerReachRandom-v0",
            "MyoHandPoseFixed":                 "myoHandPoseFixed-v0",
            "MyoHandPoseRandom":                "myoHandPoseRandom-v0",
            "MyoHandReachFixed":                "myoHandReachFixed-v0",
            "MyoHandReachRandom":               "myoHandReachRandom-v0",
            "MyoElbowPoseFixed":                "myoElbowPose1D6MFixed-v0",
            "MyoElbowPoseRandom":               "myoElbowPose1D6MRandom-v0",
            "MyoHandKeyTurnFixed":              "myoHandKeyTurnFixed-v0",
            "MyoHandKeyTurnRandom":             "myoHandKeyTurnRandom-v0",
            "MyoBaodingBallsP1":                "myoChallengeBaodingP1-v1",
            "CustomMyoBaodingBallsP1":          "CustomMyoChallengeBaodingP1-v1",
            "CustomMyoReorientP1":              "CustomMyoChallengeDieReorientP1-v0",
            "CustomMyoReorientP2":              "CustomMyoChallengeDieReorientP2-v0",
            "MyoBaodingBallsP2":                "myoChallengeBaodingP2-v1",
            "CustomMyoBaodingBallsP2":          "CustomMyoChallengeBaodingP2-v1",
            "MixtureModelBaodingEnv":           "MixtureModelBaoding-v1",
            "CustomMyoElbowPoseFixed":          "CustomMyoElbowPoseFixed-v0",
            "CustomMyoElbowPoseRandom":         "CustomMyoElbowPoseRandom-v0",
            "CustomMyoFingerPoseFixed":         "CustomMyoFingerPoseFixed-v0",
            "CustomMyoFingerPoseRandom":        "CustomMyoFingerPoseRandom-v0",
            "CustomMyoHandPoseFixed":           "CustomMyoHandPoseFixed-v0",
            "CustomMyoHandPoseRandom":          "CustomMyoHandPoseRandom-v0",
            "CustomMyoPenTwirlRandom":          "CustomMyoHandPenTwirlRandom-v0",
            "CustomChaseTag":                   "CustomChaseTagEnv-v0",
            'CustomRelocateEnv':                "CustomRelocateEnv-v0",
            'CustomRelocateEnvPhase2':          "CustomRelocateEnvPhase2-v0",
            "RelocateEnvPhase2":                "myoChallengeRelocateP2-v0",
            "ChaseTagEnvPhase2":                "myoChallengeChaseTagP2-v0",
            "Bimanual":                         "myoChallengeBimanual-v0",
            "MuscleElbowPoseFixed":             "MuscleElbowPoseFixed-v0",
            "MuscleElbowPoseRandom":            "MuscleElbowPoseRandom-v0",
            "MuscleFingerPoseFixed":            "MuscleFingerPoseFixed-v0",
            "MuscleFingerPoseRandom":           "MuscleFingerPoseRandom-v0",
            "MuscleHandPoseFixed":              "MuscleHandPoseFixed-v0",
            "MuscleHandPoseRandom":             "MuscleHandPoseRandom-v0",
            "MuscleHandPoseRandomHalfRange":    "MuscleHandPoseRandomHalfRange-v0",
            "MuscleFingerReachFixed":           "MuscleFingerReachFixed-v0",
            "MuscleFingerReachRandom":          "MuscleFingerReachRandom-v0",
            "MuscleHandReachFixed":             "MuscleHandReachFixed-v0",
            "MuscleHandReachRandom":            "MuscleHandReachRandom-v0",
            "MuscleBaodingEnvP0":               "MuscleBaodingP0-v1",
            "MuscleBaodingEnvP1":               "MuscleBaodingP1-v1",
            "MuscleBaodingEnvP2":               "MuscleBaodingP2-v1",
            "MuscleBaodingEnvP3":               "MuscleBaodingP3-v1",
            "MuscleReorientEnvP0":              "MuscleDieReorientP0-v0",
            "MuscleReorientEnvP1":              "MuscleDieReorientP1-v0",
            "MuscleReorientEnvP2":              "MuscleDieReorientP2-v0",
            "MuscleLegsStandEnv":               "MuscleLegDemo-v0",
            "MuscleLegsWalkEnv":                "MuscleLegWalk-v0",
            "WalkerBulletEnv":                  "Walker2DBulletEnv-v0",
            "HalfCheetahBulletEnv":             "HalfCheetahBulletEnv-v0",
            "AntBulletEnv":                     "AntBulletEnv-v0",
            "HopperBulletEnv":                  "HopperBulletEnv-v0",
            "HumanoidBulletEnv":                "HumanoidBulletEnv-v0",
            "HumanoidFlagrunBulletEnv":         "HumanoidFlagrunBulletEnv-v0",
            "HumanoidFlagrunHarderBulletEnv":   "HumanoidFlagrunHarderBulletEnv-v0",
        }

        if env_name in env_map:
            return gym.make(env_map[env_name], **kwargs)
        else:
            raise ValueError("Environment name not recognized:", env_name)