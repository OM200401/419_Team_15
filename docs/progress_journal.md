# Progress Log

| Week                        | Events                              | Deliverables                                                  |
|-----------------------------|-------------------------------------|---------------------------------------------------------------|
| **Week 5 (02/01 - 02/07)**  | [Team Meeting #1](#02-02-2026)<br/> |                                                               |
| **Week 6 (02/08 - 02/14)**  |                                     |                                                               |
| **Week 7 (02/15 - 02/21)**  |                                     |                                                               | 
| **Week 8 (02/22 - 02/28)**  |                                     | Project Proposal (02/27)<br/>Progress Journal (02/27)         |
| **Week 9 (03/01 - 03/07)**  |                                     |                                                               |
| **Week 10 (03/08 - 03/14)** |                                     |                                                               |
| **Week 11 (03/15 - 03/21)** |                                     |                                                               |
| **Week 12 (03/22 - 03/28)** |                                     | EvalAI Model Performance (03/26)<br/>Progress Journal (03/26) |



---

<details open id="02-02-2026">
<summary>
<h2 style="display:inline">Monday, 02/02/2026</h2>
<h3 style="display:inline">Team Meeting</h3>
</summary>

---

### Summary 
* Discussed project proposal & associated tasks
* Set a timeline for members to have the repository running on their own environment
* Set a plan to begin the progress journal and documentation

### Attendance

| Member                | Attendance (Present/Absent) |
|-----------------------|-----------------------------|
| James Birnie          | Absent                      | 
| **Milan Bertolutti**  | **Present**                 |
| Kelvin Chen           | Absent                      |
| **Bridgette Hunt**    | **Present**                 |
| **Om Mistry**         | **Present**                 |
| **Karim Jassani**     | **Present**                 |

### Minutes

* Everyone gets their environment setup and reproduces results from the research papers
  * You can create your own branch to do the above
* Bridgette is creating a docs branch where we can update the progress
* We meet again on Friday where everyone has completed reproducing the results and we can continue with whoever was able to produce the best results
* Went over project proposal template to form expectations for necessary work in the next few weeks

</details>

---

<details open id="02-06-2026">
<summary>
<h2 style="display:inline">Friday, 02/06/2026</h2>
<h3 style="display:inline">Team Meeting</h3>
</summary>

---

### Summary 
* Discussed current progress on reproducing results from research papers
* Discussed further steps on the project proposal
* Decided on meeting time for a weekly meeting on **Tuesday 11:30 am** starting next week


### Attendance

| Member                | Attendance (Present/Absent) |
|-----------------------|-----------------------------|
| **James Birnie**      | **Present**                 | 
| **Milan Bertolutti**  | **Present**                 |
| **Kelvin Chen**       | **Present**                 |
| **Bridgette Hunt**    | **Present**                 |
| **Om Mistry**         | **Present**                 |
| **Karim Jassani**     | **Present**                 |

### Minutes

* Om was able to reproduce results from the second research paper but the test accuracy was only 29.31% 
* Everyone else tried to clone the provided repo from the first research paper but had issues with the environment setup and reproducing results
* Bridgette is currently working on setting up the environment for the first research paper on her local machine
* James is going to start working on the Literature Review section of the project proposal
* Once Bridgette has the baseline model up and running, we will all set it up and then dicuss further gaps to improve performance and accuracy
* Next meeting on Tuesday to discuss progress

</details>

---

<details open id="02-10-2026">
<summary>
<h2 style="display:inline">Tuesday, 02/10/2026</h2>
<h3 style="display:inline">Team Meeting</h3>
</summary>

---

### Summary 
* Base pipeline has been successfully set up and tested locally
* Reviewed current project proposal status and identified gaps
* Discussed model improvements and next steps for the project
* Created detailed action breakdown for remaining proposal sections

### Attendance

| Member                | Attendance (Present/Absent) |
|-----------------------|-----------------------------|
| James Birnie          | Present                     | 
| Milan Bertolutti      | Present                     |
| Kelvin Chen           | Present                     |
| Bridgette Hunt        | Present                     |
| Om Mistry             | Present                     |
| Karim Jassani         | Present                     |

### Minutes

#### Current Progress
* Bridgette successfully set up the baseline Koshkina et al. pipeline and reproduced results locally
* Om tested the legibility classifier: achieved 46.7% legible predictions on test sample (7/15 images) with well-calibrated confidence scores (0.0002–0.9997), showing good separation between legible and illegible players
* Pipeline setup remains difficult on other machines; team goal is everyone running baseline by end of week

#### Proposal Sections & Assignments
* **Section 2.1** (Literature Review): James
* **Section 2.2** (Replication Results): Bridgette
* **Section 3.1** (Proposed Approach): Om, Karim, Milan (coordinating different components)
* **Section 3.2** (Model Architecture)
* **Section 3.3** (Data Preprocessing)
* **Section 4.1 & 4.2** (Enhancements & Justification)

#### Key Gaps & Improvement Strategy
**Critical Bottlenecks:**
* Pipeline reproducibility: difficult to set up on different machines
* Keyframe selection: not currently implemented before STR, likely causing poor performance on blurry/low-quality frames
* Model optimization: ResNet34 mentioned in paper as alternative for STR

**Proposed Improvements:**
* **Option A - Keyframe Identification Module:** Pre-filter frames before STR to improve accuracy (Om leading)
* **Option B - STR Architecture Enhancement:** Explore ResNet34 for STR improvements (Milan investigating)
* **Option C - Other Techniques:** To be identified through further research (Karim investigating)

#### Action Items for Next Week
* [ ] **Bridgette:** Complete Section 2.2—document baseline model results, inference times, and bottlenecks
* [ ] **James:** Complete Section 2.1—summarize key research papers and findings
* [ ] **Om:** Design keyframe identification module; update Sections 3.1 & 3.2
* [ ] **Milan:** Investigate ResNet34 for STR; update Sections 3.1 & 3.2
* [ ] **Karim:** Research additional improvement techniques; update Sections 3.1 & 3.3
* [ ] **Kelvin:** Evaluate dockerization to improve reproducibility across machines
* [ ] **All:** Get baseline pipeline running locally by end of week

</details>

---