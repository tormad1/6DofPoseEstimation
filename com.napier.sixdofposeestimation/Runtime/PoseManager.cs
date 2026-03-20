using UnityEngine;
using System;

public class PoseManager : MonoBehaviour
{
    [Header("Target")]
    public Transform target;

    [Header("Tracking")]
    public float confidenceThreshold = 0.6f;
    public float trackingTimeoutSeconds = 0.5f;

    [Header("Jump Detection")]
    public float maxPositionDelta = 0.5f;
    public float maxRotationDeltaDeg = 45f;
    public float candidateConfirmDistance = 0.3f;
    public float candidateDirectionDot = 0.6f;

    [Header("Hardcoded Pose (for testing)")]
    public Vector3 hardcodedPosition = new Vector3(0, 1, 2);
    public Vector3 hardcodedEulerRotation = new Vector3(0, 45, 0);
    public float hardcodedConfidence = 0.9f;

    public bool TrackingLost { get; private set; } = true;

    struct PoseData
    {
        public Vector3 position;
        public Quaternion rotation;
        public float confidence;
        public double timestamp;
    }

    PoseData? lastAcceptedPose = null;
    PoseData? candidatePose = null;

    double lastAcceptedTimestamp = double.MinValue;
    float lastAcceptTimeUnity = 0f;

    void Update()
    {
        if (TryGetHardcodedPose(out PoseData pose))
        {
            ProcessIncomingPose(pose);
        }

        // Timeout handling
        if (!TrackingLost &&
            Time.time - lastAcceptTimeUnity > trackingTimeoutSeconds)
        {
            TrackingLost = true;
        }
    }

    bool TryGetHardcodedPose(out PoseData pose)
    {
        pose = new PoseData
        {
            position = hardcodedPosition,
            rotation = Quaternion.Euler(hardcodedEulerRotation),
            confidence = hardcodedConfidence,
            timestamp = Time.timeAsDouble
        };

        return true;
    }

    void ProcessIncomingPose(PoseData pose)
    {
        if (pose.timestamp <= lastAcceptedTimestamp)
            return;

        if (!IsPoseValid(pose))
            return;

        if (pose.confidence < confidenceThreshold)
        {
            SetTrackingLost();
            return;
        }

        if (!lastAcceptedPose.HasValue)
        {
            AcceptPose(pose);
            return;
        }

        if (IsBigJump(lastAcceptedPose.Value, pose))
        {
            HandleCandidate(pose);
            return;
        }

        AcceptPose(pose);
    }

    void HandleCandidate(PoseData newPose)
    {
        if (!candidatePose.HasValue)
        {
            candidatePose = newPose;
            return;
        }

        PoseData candidate = candidatePose.Value;

        float distToOld = Vector3.Distance(lastAcceptedPose.Value.position, newPose.position);
        float distToCandidate = Vector3.Distance(candidate.position, newPose.position);

        Vector3 dirOldToCandidate =
            (candidate.position - lastAcceptedPose.Value.position).normalized;

        Vector3 dirCandidateToNew =
            (newPose.position - candidate.position).normalized;

        float dirDot = Vector3.Dot(dirOldToCandidate, dirCandidateToNew);

        bool nearOld = distToOld < candidateConfirmDistance;
        bool nearCandidate = distToCandidate < candidateConfirmDistance;
        bool continuesDirection = dirDot > candidateDirectionDot;

        if (nearOld)
        {
            candidatePose = null;
            return;
        }

        if (nearCandidate || continuesDirection)
        {
            AcceptPose(newPose);
            candidatePose = null;
            return;
        }

        candidatePose = newPose;
    }

    void AcceptPose(PoseData pose)
    {
        lastAcceptedPose = pose;
        lastAcceptedTimestamp = pose.timestamp;
        lastAcceptTimeUnity = Time.time;

        candidatePose = null;

        TrackingLost = false;

        ApplyToTarget(pose);
    }

    void ApplyToTarget(PoseData pose)
    {
        if (target == null) return;

        target.position = pose.position;
        target.rotation = pose.rotation;
    }

    void SetTrackingLost()
    {
        TrackingLost = true;
    }

    bool IsPoseValid(PoseData p)
    {
        if (!IsFiniteVector(p.position))
            return false;

        if (!IsFiniteQuaternion(p.rotation))
            return false;

        return true;
    }

    bool IsFiniteVector(Vector3 v)
    {
        return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
    }

    bool IsFiniteQuaternion(Quaternion q)
    {
        if (!IsFinite(q.x) || !IsFinite(q.y) ||
            !IsFinite(q.z) || !IsFinite(q.w))
            return false;

        // Check quaternion is approximately normalized
        float sqrMag = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        return Mathf.Abs(1f - sqrMag) < 0.01f;
    }

    bool IsFinite(float f)
    {
        return !(float.IsNaN(f) || float.IsInfinity(f));
    }

    bool IsBigJump(PoseData oldPose, PoseData newPose)
    {
        float posDelta = Vector3.Distance(oldPose.position, newPose.position);
        float rotDelta = Quaternion.Angle(oldPose.rotation, newPose.rotation);

        return posDelta > maxPositionDelta || rotDelta > maxRotationDeltaDeg;
    }
}