# Lab Deliverables Index

Per Lab 5 manual section 3.2.3 ("Prepare an index file for easy access of your lab
deliverables"). All five labs' deliverables live in their own `lab1/`-`lab5/` folders at the
repo root; this file is the single entry point into all of them. All documents are PDF, all
diagrams are JPG, except `lab5/README.md` (kept as Markdown) and `lab3/ai_skeleton_output.zip`
(a zip, as Lab 3's manual requires).

## Lab 1 -- Requirements and Initial Use Case Model

| File | What it is |
|---|---|
| [`lab1/requirements.pdf`](lab1/requirements.pdf) | Functional and non-functional requirements specification v1.0. |
| [`lab1/data_dictionary.pdf`](lab1/data_dictionary.pdf) | Data dictionary. |
| [`lab1/use_case_model.pdf`](lab1/use_case_model.pdf) | Initial Use Case Model: diagram + descriptions, v1.0. |
| [`lab1/use_case_diagram.jpg`](lab1/use_case_diagram.jpg) | Use case diagram (image). |
| [`lab1/ui_mockups.pdf`](lab1/ui_mockups.pdf) + [`lab1/ui_mockups/`](lab1/ui_mockups/) | UI mockups, compiled PDF and individual screen images. |
| [`lab1/ai_critique_report.pdf`](lab1/ai_critique_report.pdf) | Report on the AI's critique of the Use Case diagram and requirements. |

## Lab 2 -- Refined Use Case Model and Design Model

| File | What it is |
|---|---|
| [`lab2/use_case_descriptions.pdf`](lab2/use_case_descriptions.pdf) | Refined Use Case descriptions, v1.1. |
| [`lab2/use_case_diagram.jpg`](lab2/use_case_diagram.jpg) | Refined Use Case diagram (image). |
| [`lab2/entity_classes.jpg`](lab2/entity_classes.jpg) | Class diagram of entity classes (plain data records: `HeadPose`, `WindowBounds`, `ContentBlock`, `CalibrationSession`, `SessionMeta`, `AffineCorrection`). |
| [`lab2/boundary_control_classes.jpg`](lab2/boundary_control_classes.jpg) | Class diagram of the key boundary classes (OS/hardware interfaces, e.g. `App`, `ScrollInjector`, `WindowBoundsProvider`) and control classes (decision/coordination logic, e.g. `ScrollTrigger`, `AttentionGate`, `CommandRunner`). |
| [`lab2/dialog_map.pdf`](lab2/dialog_map.pdf) / [`lab2/dialog_map.jpg`](lab2/dialog_map.jpg) | Initial dialog map. |
| [`lab2/tech_stack_report.pdf`](lab2/tech_stack_report.pdf) | Report on the AI's technology-stack recommendation. |
| [`lab2/conceptual_model.pdf`](lab2/conceptual_model.pdf) | Conceptual model (supporting document, carried from the team's own Lab 2 work). |

## Lab 3 -- Design Model and System Architecture

| File | What it is |
|---|---|
| [`lab3/use_case_model.pdf`](lab3/use_case_model.pdf) | Use Case Model, carried forward (v1.2), + [`lab3/use_case_diagram.jpg`](lab3/use_case_diagram.jpg). |
| [`lab3/class_diagram.jpg`](lab3/class_diagram.jpg) | Design Model: class diagram (reused from the Lab 5 asset set, the single unified diagram this class diagram is split from for Lab 2). |
| [`lab3/dialog_map.jpg`](lab3/dialog_map.jpg) | Design Model: dialog map. |
| [`lab3/system_architecture.pdf`](lab3/system_architecture.pdf) | System architecture description (layers, design patterns actually used, persistent data, access control). |
| [`lab3/ai_project_init_report.pdf`](lab3/ai_project_init_report.pdf) | Report on the AI-based project-initialisation exercise: the repeated technology-stack prompt (3.3.1-3.3.2) and the AI-generated application-skeleton exercise (3.3.3-3.3.4), with prompts, AI responses, and critique. |
| [`lab3/ai_skeleton_output.zip`](lab3/ai_skeleton_output.zip) | The minimal skeleton (folders/TODO-only files) the AI generated in that exercise -- distinct from the application's actual code under `src/`. |

## Lab 4 -- Testing and Coding Agent Exercises

| File | What it is |
|---|---|
| [`lab4/test_cases.pdf`](lab4/test_cases.pdf) | Test case design and results (black-box `TC-ST-*`, white-box `TC-CGT-*`/`TC-AG-*`). |
| [`lab4/demo_script.pdf`](lab4/demo_script.pdf) | Demo script. |
| [`lab4/coding_agent_report.pdf`](lab4/coding_agent_report.pdf) | Report on the coding-agent exercises: deleting and regenerating a control class from scratch, plus one further experiment, each with a reflection on approach, materials given, and outcome. |

## Lab 5 -- Final Documentation and Demo

| File | What it is |
|---|---|
| [`lab5/README.md`](lab5/README.md) | What changed and why, relative to Labs 1-2 (the full rationale for every correction below). |
| [`lab5/srs.pdf`](lab5/srs.pdf) | Software Requirements Specification v2.0, with a "Changes Since Lab 2" section. |
| [`lab5/use_case_model.pdf`](lab5/use_case_model.pdf) / [`lab5/use_case_diagram.jpg`](lab5/use_case_diagram.jpg) | Final Use Case Model v2.0. |
| [`lab5/class_diagram.jpg`](lab5/class_diagram.jpg) | Final class diagram. |
| [`lab5/dialog_map.jpg`](lab5/dialog_map.jpg) | Final dialog map / UI state machine. |

The demo video and peer review surveys are submitted separately (outside this repo) per the Lab 5
manual and are not listed here.
