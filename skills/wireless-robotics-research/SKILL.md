---
name: wireless-robotics-research
description: Research guidance for wireless robotics, including system modeling, method development, and experiments involving perception, communication, computation, and action.
---

# Wireless Robotics Research

*Public edition, adapted from the original WirelessRoboAgent research skill.*

Develop well-founded, effective, and reproducible approaches to wireless robotics problems. Relate the system model, design choices, and experimental evidence to the application's objectives and operating conditions.

## Understanding the System

- Identify the research question, operating conditions, objectives, and constraints. Clarify priorities and trade-offs relevant to robot behavior and resource use.

- Describe what each decision-maker can know and when it can act. Distinguish physical state, observations, estimates, and received information. Account for the timing of information acquisition, message delivery, computation, and action, including the order of decisions, state updates, and feedback.

- Examine how communication and robotic processes influence one another. Communication affects information availability and timeliness, while robot behavior can change transmission conditions and information needs. Shared resources, delayed effects, and dependencies between decisions can influence future feasibility and overall outcomes.

- Choose a system model that captures the relevant relationships at an appropriate level of detail. Define variables, index domains, units, state ownership, updates, and resource accounting clearly. Distinguish supplied definitions from introduced assumptions, and resolve symbols and references in context. Keep hard availability and safety conditions explicit when describing feasible behavior.

- Consult relevant papers, primary technical sources, and official implementations where they clarify modeling or design choices. Check whether their assumptions fit the system. `sub_agent` can help with focused analysis or review; assess its findings against relevant sources and evidence.

## Method Development and Refinement

- Choose an approach suited to the problem structure, available information, and computational resources. Explain its main trade-offs and how its decision horizon and coordination scope relate to the system dynamics.

- Describe the algorithm's inputs, maintained state, decision rules, updates, constraint handling, and important parameters. Ground consequential choices and approximations in system scales, analytical reasoning, or supporting evidence.

- Carry the model's information boundaries, event ordering, and resource constraints into the implementation. Keep system dynamics, decision logic, and measurements clear enough to inspect separately.

- Small cases with known outcomes can help check timing, information availability, and resource accounting. Check measurements against the states and events they describe.

- Use execution traces and reference comparisons to investigate unexpected behavior and distinguish parameter calibration from limitations of the decision mechanism. Let the findings guide revisions, comparing benefits, regressions, and resource costs under matched conditions and rechecking affected behavior.

## Experimental Design, Evaluation, and Interpretation

- Design experiments around the research questions and intended claims, choosing operating conditions and measurements that distinguish relevant explanations or design choices.

- Compare with established methods or justified references under comparable conditions, information access, and resource budgets. Check that each reference behaves as intended.

- Use ablations or sensitivity studies where they help explain a mechanism or its dependence on operating conditions. Choose interventions that exercise the mechanism or relationship being examined.

- Retain the configurations, measurement definitions, and outcomes behind each comparison. Use suitable repetitions and uncertainty estimates when variability affects the conclusions.

- Interpret results in terms of system behavior, benefits, costs, and trade-offs. Distinguish measured effects from analytical estimates or proposed explanations. Discuss contrary evidence, material limitations, and the conditions supporting the conclusions.
