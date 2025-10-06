# Technical Learning Query Templates

**Use these templates when asking about algorithms, concepts, or technical topics.**

---

## TEMPLATE 1: Learning New Algorithm

```
I need to understand and implement [ALGORITHM NAME].

CONTEXT:
- Current level: [beginner/intermediate/advanced] in [domain]
- Prerequisites I have: [list what you know]
- Goal: [what you want to achieve with this]

REQUIREMENTS:
1. Pattern identification: What family does this algorithm belong to?
2. Visual diagram: Show how it works step-by-step
3. Mathematical intuition: Explain the math with real-world analogy
4. Implementation: 3 versions (basic/practical/optimized)
5. Decision framework: When to use vs alternatives
6. Common gotchas: What mistakes do beginners make?

PROJECT:
Guide me to implement this from scratch in [timeframe].
Success criteria: [how I'll know I understand it]

FORMAT:
- Pattern-first explanation
- Heavily annotated code
- Decision trees for usage
- Clear next steps
```

**Example:**
```
I need to understand and implement Gradient Boosting.

CONTEXT:
- Current level: Understand decision trees and basic ensemble methods
- Prerequisites: Know bagging, random forests, gradient descent
- Goal: Compete in Kaggle competition, need advanced methods

REQUIREMENTS:
1. Pattern: How does boosting differ from bagging conceptually?
2. Visual: Show how trees are built sequentially
3. Math: Explain gradient descent in function space (simply)
4. Implementation: NumPy version, then sklearn, then XGBoost
5. Decision: When to use GBM vs Random Forest vs Neural Net?
6. Gotchas: Overfitting, learning rate, tree depth issues

PROJECT:
Implement basic gradient boosting in 1 week.
Success: Beat Random Forest on Kaggle dataset by 5%

FORMAT: Pattern-first, code examples, decision framework
```

---

## TEMPLATE 2: Learning New Concept

```
Explain [CONCEPT] for deep understanding.

CONTEXT:
- Why I'm learning this: [specific reason]
- What I already know: [related concepts]
- Confusion points: [what's unclear from other sources]

PROVIDE:
1. Analogy: Real-world comparison I can visualize
2. Pattern: Where does this concept fit in bigger picture?
3. Visual: Diagram showing relationships
4. Math: If applicable, with intuition first
5. Code: How to use/implement in practice
6. Recognition: How to identify when I need this

ANTI-PATTERNS:
- Don't: [what NOT to do with this concept]
- Common mistake: [what people get wrong]

PRACTICE:
Small project to cement understanding: [concrete suggestion]
```

**Example:**
```
Explain Backpropagation for deep understanding.

CONTEXT:
- Why: Implementing neural network from scratch
- Know: Forward pass, gradient descent, chain rule basics
- Confused: How gradients flow through layers

PROVIDE:
1. Analogy: Like tracing blame for mistake backward through pipeline
2. Pattern: This is just chain rule applied recursively
3. Visual: Computational graph with gradient flow
4. Math: Show chain rule, then generalize to layers
5. Code: Step-by-step implementation for 3-layer network
6. Recognition: Need this whenever training neural networks

ANTI-PATTERNS:
- Don't: Compute all gradients at once (memory explosion)
- Mistake: Forgetting to zero gradients between batches

PRACTICE:
Build 3-layer network, verify gradients with numerical gradient checking
```

---

## TEMPLATE 3: Implementing Paper/Technique

```
I want to implement [PAPER/TECHNIQUE NAME].

PAPER CONTEXT:
- Paper: [title/link]
- Main contribution: [what's novel]
- My understanding: [what I think it does]
- Stuck on: [specific part that's unclear]

IMPLEMENTATION HELP:
1. Pattern: What existing patterns does this extend/modify?
2. Key insight: What's the core idea (simply)?
3. Pseudocode: High-level algorithm steps
4. Critical details: What parts are essential vs optional?
5. Code skeleton: Outline with TODOs
6. Validation: How to verify it's working correctly?

SIMPLIFICATIONS:
- Can I start with simpler version?
- What can I skip for first implementation?
- Minimum for reproduction of key results?

TIMELINE:
Implement in [timeframe] with [hours available]
```

**Example:**
```
I want to implement Attention Mechanism from "Attention Is All You Need".

PAPER CONTEXT:
- Paper: Vaswani et al., 2017
- Main contribution: Self-attention for sequence modeling
- Understanding: Uses queries, keys, values for weighted combinations
- Stuck on: Multi-head attention and positional encoding

IMPLEMENTATION HELP:
1. Pattern: This is weighted averaging based on similarity
2. Key insight: Let model learn what to pay attention to
3. Pseudocode: Compute similarity → softmax → weighted sum
4. Critical: Scaled dot-product, multi-head, residual connections
5. Code skeleton: PyTorch implementation outline
6. Validation: Test on small sequence, check output shapes

SIMPLIFICATIONS:
- Start with single-head attention
- Skip positional encoding initially
- Minimum: Working attention layer that improves over baseline

TIMELINE: 1 week, 20 hours available
```

---

## TEMPLATE 4: Debugging Technical Issue

```
I'm stuck debugging [ISSUE].

PROBLEM DESCRIPTION:
- What I'm trying to do: [goal]
- What's happening: [current behavior]
- What should happen: [expected behavior]
- Error message: [if any]

CODE:
```
[paste relevant code]
```

CONTEXT:
- Dataset: [description/size]
- Environment: [Python version, libraries, hardware]
- What I've tried: [debugging steps already taken]

HELP NEEDED:
1. Root cause analysis: Why is this happening?
2. Systematic debugging approach: Step-by-step what to check
3. Quick fix: Immediate solution (even if hacky)
4. Proper fix: Right way to solve this
5. Prevention: How to avoid this in future

PATTERN:
Is this a common pattern of bug? Where else might this occur?
```

---

## TEMPLATE 5: Tool/Framework Selection

```
I need to choose between [OPTION A] and [OPTION B] for [USE CASE].

PROJECT CONTEXT:
- Building: [what kind of system]
- Requirements: [specific needs]
- Constraints: [time, resources, skills]
- Scale: [users/data/traffic expected]

COMPARISON NEEDED:
1. Feature comparison: What does each offer?
2. Learning curve: How hard to learn each?
3. Performance: Speed, memory, scalability
4. Ecosystem: Libraries, community, resources
5. Production: Deployment, monitoring, maintenance

DECISION FRAMEWORK:
Create decision tree: When to use A vs B vs [other option]

SPECIFIC QUESTIONS:
- [Question about A]
- [Question about B]
- [Question about integration]

MY SITUATION:
[Specific constraints that might affect choice]
```

**Example:**
```
I need to choose between PyTorch and TensorFlow for computer vision project.

PROJECT CONTEXT:
- Building: Image classification system for production
- Requirements: Fast iteration, good debugging, deploy to mobile
- Constraints: 3 months timeline, solo developer
- Scale: 100K images, 1K predictions/day

COMPARISON NEEDED:
1. Features: Which has better CV tools?
2. Learning: Coming from NumPy, which is easier?
3. Performance: Inference speed for mobile deployment
4. Ecosystem: Pretrained models, tutorials, community
5. Production: Deployment options, monitoring tools

DECISION FRAMEWORK:
- If prioritizing research/iteration: ?
- If prioritizing production/deployment: ?
- If prioritizing mobile: ?

SPECIFIC QUESTIONS:
- PyTorch: How hard is mobile deployment?
- TensorFlow: Is eager execution as good as PyTorch?
- Both: Which has better pretrained ResNet/EfficientNet?

MY SITUATION:
Need to iterate fast initially, deploy to Android eventually, limited DevOps skills.
```

---

## TEMPLATE 6: System Design Question

```
How do I design [SYSTEM/ARCHITECTURE]?

SYSTEM REQUIREMENTS:
- Purpose: [what it needs to do]
- Scale: [users/data/throughput]
- Constraints: [latency, cost, hardware]
- Non-functional: [reliability, maintainability]

CURRENT UNDERSTANDING:
- Components I think I need: [list]
- Uncertain about: [specific questions]
- Concerns: [potential problems]

HELP NEEDED:
1. Architecture diagram: Visual overview of components
2. Component breakdown: What each part does
3. Data flow: How data moves through system
4. Technology choices: What to use for each component
5. Scalability plan: How to grow from small to large
6. Trade-offs: What I'm optimizing for vs sacrificing

PATTERNS:
- What architecture pattern(s) apply here?
- Similar systems I can learn from?

VALIDATION:
How to validate design before building?
```

**Example:**
```
How do I design a real-time ML prediction API?

SYSTEM REQUIREMENTS:
- Purpose: Serve ML model predictions via REST API
- Scale: 1000 requests/sec, sub-100ms latency
- Constraints: Cost-effective, single cloud provider
- Non-functional: 99.9% uptime, easy to update models

CURRENT UNDERSTANDING:
- Need: API server, model serving, caching, monitoring
- Uncertain: How to handle model updates without downtime
- Concerns: Cold start latency, model versioning, A/B testing

HELP NEEDED:
1. Architecture: Diagram with load balancer → API → model server
2. Components: FastAPI? TorchServe? Redis? Prometheus?
3. Data flow: Request → preprocessing → prediction → response
4. Tech choices: Why each technology for each component
5. Scaling: Start with 10 req/sec, grow to 1000 req/sec
6. Trade-offs: Latency vs cost vs complexity

PATTERNS:
- This is microservices pattern?
- Similar: Netflix, Uber ML serving?

VALIDATION:
Load testing strategy, monitoring key metrics
```

---

## TEMPLATE 7: Math Concept Deep Dive

```
I need to understand [MATH CONCEPT] deeply for ML.

CURRENT KNOWLEDGE:
- Math background: [e.g., calculus, linear algebra]
- Why I need this: [specific ML application]
- Current confusion: [what's not clicking]

EXPLAIN WITH:
1. Intuition FIRST: Real-world analogy
2. Geometric interpretation: Visual understanding
3. Formal definition: Mathematical notation
4. Computational approach: How to calculate
5. Code implementation: NumPy/PyTorch code
6. ML application: Where this appears in ML

PROGRESSION:
- Start with: [simplest case]
- Build to: [full complexity]
- Show: [edge cases and special scenarios]

PRACTICE PROBLEM:
Concrete problem I can solve to test understanding
```

**Example:**
```
I need to understand Eigenvalues/Eigenvectors deeply for ML.

CURRENT KNOWLEDGE:
- Math: Know matrix multiplication, linear transformations
- Need for: PCA, understanding covariance matrices
- Confusion: What eigenvectors "mean" intuitively

EXPLAIN WITH:
1. Intuition: Directions that matrix only stretches (not rotates)
2. Geometric: Show transformation that preserves direction
3. Formal: Av = λv definition
4. Computational: Power iteration or np.linalg.eig
5. Code: Compute and visualize on 2x2 matrix
6. ML application: PCA finds eigenvectors of covariance matrix

PROGRESSION:
- Start: 2x2 matrix, symmetric, positive definite
- Build: Non-symmetric matrices, complex eigenvalues
- Show: Degenerate cases, repeated eigenvalues

PRACTICE:
Implement PCA from scratch using eigenvalue decomposition
```

---

## TEMPLATE 8: Project Planning

```
I want to build [PROJECT] to learn [SKILLS].

PROJECT IDEA:
- Description: [what you're building]
- Complexity level: [beginner/intermediate/advanced]
- Time available: [hours per week, total weeks]
- End goal: [what constitutes "done"]

LEARNING OBJECTIVES:
- Skills to acquire: [list specific skills]
- Concepts to understand: [theoretical knowledge]
- Tools to master: [frameworks, libraries]

STRUCTURE NEEDED:
1. Milestone breakdown: Week-by-week plan
2. Dependencies: What must I learn first?
3. Scope definition: MVP vs nice-to-have features
4. Success metrics: How to measure progress?
5. Gotchas: Known issues with this type of project?
6. Reference implementations: Similar projects to learn from?

ADHD OPTIMIZATION:
- Weekly shipping: What can I demo each Friday?
- Hyperfocus sessions: Best components for deep dives?
- Context switching: Alternative tasks when stuck?

VALIDATION:
How do I know I'm on track? What metrics matter?
```

**Example:**
```
I want to build an image style transfer app to learn CNNs.

PROJECT IDEA:
- Description: Upload photo, apply artistic style (e.g., Van Gogh)
- Complexity: Intermediate
- Time: 10 hours/week for 4 weeks
- End goal: Web app with 3+ styles, sub-5sec inference

LEARNING OBJECTIVES:
- Skills: CNN architectures, transfer learning, model deployment
- Concepts: Feature extraction, style loss, content loss
- Tools: PyTorch, pretrained VGG, FastAPI, Streamlit

STRUCTURE NEEDED:
1. Milestones:
   - Week 1: Understand algorithm, basic implementation
   - Week 2: Optimize inference, add multiple styles
   - Week 3: Build web interface
   - Week 4: Deploy and polish
2. Dependencies: CNN basics → Transfer learning → Style transfer
3. Scope: MVP = 1 style working; Nice = multiple styles, fast inference
4. Success: Working demo I can share on Twitter
5. Gotchas: Slow inference, memory issues, style quality
6. References: fast.ai style transfer, PyTorch tutorials

ADHD OPTIMIZATION:
- Week 1: Ship style transfer on single image
- Week 2: Ship API endpoint
- Week 3: Ship web interface
- Week 4: Ship public demo

Hyperfocus: Weeks 1-2 (algorithm implementation)
Switch tasks: If stuck on optimization, work on UI

VALIDATION:
- Week 1: Generated image looks styled
- Week 2: <5 sec inference time
- Week 3: Friend can use web app
- Week 4: Posted on Twitter, got feedback
```

---

## TEMPLATE 9: Performance Optimization

```
I need to optimize [COMPONENT/SYSTEM] for [METRIC].

CURRENT STATE:
- What I'm optimizing: [code/model/system]
- Current performance: [metrics]
- Target performance: [goals]
- Bottleneck hypothesis: [what I think is slow]

PROFILING DATA:
[Include profiling output if you have it]

OPTIMIZATION HELP:
1. Profiling: What to measure first?
2. Bottleneck identification: Systematic approach
3. Quick wins: Easy optimizations (80/20 rule)
4. Deep optimizations: More complex improvements
5. Trade-offs: Performance vs readability/maintainability
6. Validation: How to ensure correctness after optimization?

CONSTRAINTS:
- Can't change: [fixed requirements]
- Willing to trade: [acceptable sacrifices]
- Resources available: [hardware, time]

PATTERN:
Common optimization patterns for this type of problem?
```

---

## TEMPLATE 10: Career/Learning Path

```
I want to become [ROLE/EXPERT] in [TIMEFRAME].

CURRENT STATE:
- Experience: [current level]
- Skills I have: [existing knowledge]
- Time commitment: [hours per week]
- Learning style: [preferences from profile]

TARGET STATE:
- Goal role: [specific position or expertise level]
- Must-have skills: [non-negotiable competencies]
- Nice-to-have skills: [bonus competencies]
- Portfolio requirements: [what to demonstrate]

PATH NEEDED:
1. Skill gap analysis: What am I missing?
2. Learning roadmap: Month-by-month progression
3. Project portfolio: What to build to demonstrate skills
4. Milestones: How to track progress?
5. Market validation: How to know I'm ready?

OPTIMIZATION:
- Given my cognitive profile, how to learn fastest?
- What should I focus on vs skip?
- Parallel tracks vs sequential learning?

REALISTIC ASSESSMENT:
- Is this timeline achievable?
- What could go wrong?
- Plan B if timeline slips?
```

---

## USAGE GUIDELINES

### When to Use Which Template

**Algorithm/Concept Learning:**
- Templates 1, 2, 7 → Understanding how things work

**Building Projects:**
- Templates 3, 6, 8 → Planning and implementing

**Problem Solving:**
- Templates 4, 9 → Debugging and optimizing

**Decision Making:**
- Template 5 → Choosing between options

**Career Planning:**
- Template 10 → Long-term strategy

### How to Customize Templates

1. **Fill in all [BRACKETS]** with your specific context
2. **Remove sections** that aren't relevant
3. **Add sections** for your specific needs
4. **Combine templates** when query spans multiple areas
5. **Reference your cognitive profile** for personalization

### Template Enhancement

Add these to any template for better responses:

```
COGNITIVE OPTIMIZATION:
- ADHD: [specific hyperfocus opportunities, engagement tactics]
- Autism: [need explicit rules for X, systematic framework for Y]
- Pattern: [similar to concept Z I already know]

ANTI-PATTERNS TO AVOID:
- Don't explain: [topics I already understand]
- Don't suggest: [approaches I've tried that failed]
- Don't assume: [background I don't have]
```

---

## REMEMBER

**Good query = Good response**

Elements of effective query:
1. ✅ Specific context (not vague)
2. ✅ Clear goal (what you want to achieve)
3. ✅ Explicit format request (structure needed)
4. ✅ Time constraints (realistic scope)
5. ✅ Success criteria (how you'll measure)

**Poor query:**
"Explain neural networks"

**Good query:**
"Explain neural networks using Template 2, focusing on backpropagation intuition. I know linear algebra and calculus. Goal: implement 3-layer network from scratch this weekend. Need pattern-first explanation with heavily annotated code."