# Motion Analysis Improvements

Potential improvements for the STPE motion analysis pipeline.

---

## 1. Temporal Context Captioning

By feeding previous frames' game state metadata to the captioning model, it can:

- Describe what happened **between** the previous frame and current frame
- Reason about state transitions (e.g., "character transitioned from Running to Jumping")
- Provide causal explanations for observed changes

### Example

**Frame N-1 metadata:**
```json
{
  "humanoid_state": "Running",
  "velocity": [16.0, 0, 0],
  "inputs": ["W"]
}
```

**Frame N metadata:**
```json
{
  "humanoid_state": "Jumping",
  "velocity": [16.0, 12.0, 0],
  "inputs": ["W", "Space"]
}
```

**Enhanced caption:**
> "Player pressed Space while running forward, initiating a jump. Vertical velocity increased from 0 to 12 studs/sec while maintaining forward momentum."

This temporal reasoning produces richer training data compared to single-frame captions.
