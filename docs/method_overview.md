# Method Overview

The pipeline is training-free and runs at inference time.

For each question, the runner first asks the video-language model for a direct answer from the complete video. Questions involving temporal order, spatial relations, visibility, motion, direction, or counting are routed to an evidence-refinement path. That path builds candidate answers with additional video passes, verifies candidate replacements against the direct answer, and applies a conservative safety gate before updating the final choice.

The public code keeps the complete runnable flow compact: direct prediction, routed candidate generation, repeated verification, gated replacement, and final JSON writing.
