# Social Engineering Attack Detection in Corporate Communications

**Team Members:** Pappu Jha, Hanzla Hamid, Nick Rahimi  

---
> ⚠️ This project is currently under active research and development. All implementations, models, and evaluations will be added progressively.

## Project Overview

Develop a system to detect social engineering attempts in emails, chat messages, and phone communications. Create behavioral models to identify manipulati on techniques and psychological pressure tactics. Build an employee training system that adapts based on individual vulnerability patterns. Implement real-time alerting for suspicious communication patterns.

### Overall Research Objective

To model how social engineering attacks (like phishing or manipulation) spread within an organization by analyzing internal communication networks (social graphs), identifying vulnerable roles, and predicting attack paths.

---

## Research Objectives

### Communication Analysis
- Linguistic analysis for urgency indicators and emotional manipulation
- Sender verification and relationship mapping within organizations
- Content analysis for information requests and credential harvesting attempts
- Temporal pattern analysis for coordinated social engineering campaigns

### Behavioral Modeling
- Employee susceptibility profiling based on historical interactions
- Risk scoring for different communication types and contexts
- Adaptive training recommendations based on individual vulnerability patterns
- Gamification elements for security awareness improvement

### Detection and Response
- Real-time monitoring of communication channels
- Automated alerting for high-risk interactions
- Integration with security awareness training platforms
- Incident response workflow automation for suspected social engineering

---

## Phase-Wise Research Roadmap

### Phase 1: Foundation & Understanding
**Goal:** Build a solid understanding of social engineering and trust-based graph analysis.

**Tasks:**
- Study the paper "Modeling of Social Engineering Attacks Based on Social Graph of Employees"
- Learn basics of:
  - Graph Theory (nodes, edges, centrality, shortest paths)
  - Social Engineering techniques (e.g., spear phishing, impersonation)
  - Trust graphs & influence metrics

**Resources:**
- "Social Network Analysis" by Stanley Wasserman (book)
- NetworkX documentation
- UCINET dataset samples

### Phase 2: Dataset Acquisition
**Goal:** Collect or simulate real-world communication data and attack behavior patterns.

**Tasks:**
- Acquire or simulate:
  - Internal communication data (emails, messages, etc.)
  - Role labels (e.g., HR, finance, admin, etc.)
  - (Optional) Trust levels between users or departments

**Options:**
- Use open datasets:
  - Enron Email Dataset (very commonly used)
  - Simulated phishing datasets from academic security labs
- If unavailable:
  - Create a small synthetic dataset with:
    - 50–100 employees
    - Role-based labels
    - Directed communication graph (sender → receiver)

### Phase 3: Social Graph Construction
**Goal:** Convert communication data into a network graph for analysis.

**Tasks:**
- Use NetworkX (Python) to build the graph:
  - Nodes = Employees
  - Edges = Communication frequency or trust
  - Edge weights = Message count / trust level
- Perform analysis:
  - Degree Centrality: Who communicates the most?
  - Betweenness Centrality: Who connects groups?
  - Community Detection (e.g., using Louvain method)

### Phase 4: Attack Simulation
**Goal:** Model how an attacker might navigate the organization.

**Tasks:**
- Define attacker goals (e.g., reach the CEO starting from a compromised intern)
- Apply algorithms like:
  - Dijkstra's shortest path (if attacker prefers fast)
  - Trust score-based paths (attacker uses the most trusted intermediaries)
- Simulate attack chains:
  - "Who is most likely to be exploited?"
  - "Which path leads to the most critical asset?"

### Phase 5: Identify Vulnerabilities & Insights
**Goal:** Analyze simulation results to draw conclusions.

**Tasks:**
- Highlight roles most susceptible (e.g., HR, Admin, Secretaries)
- Visualize:
  - Graph heatmaps
  - Attack path animations
- Suggest training or policy recommendations:
  - Increase awareness in vulnerable roles
  - Adjust access controls

---

## Dataset Suggestions

- **Social Engineering Dataset:** Academic collections from security research
- **Corporate Communications:** Anonymized internal communications (with consent)
- **Phishing Simulation Results:** Data from security awareness training platforms
- **Psychological Manipulation Examples:** Academic studies on influence techniques

---

## Background

Social engineering is the most powerful tool an attacker can use to access knowledge by manipulating a person into giving information. It is superior to most other forms of hacking in that it can breach even the most secure systems, as the users themselves are the most vulnerable part of the system. Social engineering has become an emerging threat in virtual communities. Research has shown that social engineering is easy to automate in many cases and can therefore be performed on a large scale.

While the awareness of social engineering in emails has increased, the awareness in cloud services and social networks is still comparatively low.
