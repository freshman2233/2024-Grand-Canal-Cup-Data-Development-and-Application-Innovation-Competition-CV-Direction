# 2024 "Grand Canal Cup" Data Development and Application Innovation Competition – CV Direction

**Language**

[English](./README.md)     [中文](./README_CN.md)

## 1. Project Overview

This project is the competition code for the **2024 "Grand Canal Cup" Data Development and Application Innovation Competition - CV Direction**.

The goal is to develop an intelligent recognition system that can automatically detect and classify violations in urban management.

The system should utilize advanced image processing and computer vision technologies to analyze videos captured by surveillance cameras, accurately identify violations, and promptly alert the management department to achieve more efficient urban governance.

The final ranking was **142/522**, achieving advanced preprocessing and hyperparameter tuning, which enhanced the **F1** and **MOTA** metrics.

**Competition Link**: https://www.marsbigdata.com/competition/details?id=3839107548872

## 2. Competition Overview

### **I. Background**

To thoroughly implement President Xi Jinping's important discussions on the protection, inheritance, and utilization of the Grand Canal culture, align with the digital China and digital Jiangsu development strategies, focus on developing new productive forces, fully stimulate the innovation vitality of data application development, and maximize the multiplier effect of data elements to empower economic and social development.

### **II. Theme and Timeline**

- **Theme:** Data Convergence in the Grand Canal, Sailing Across the World
- **Timeline:** July - September 2024

### **III. Organizational Structure**

- **Guiding Unit:** Jiangsu Provincial Data Administration
- **Organizers:** Yangzhou Municipal Government, National Information Center
- **Undertaking Units:** Jiangsu Provincial Big Data Management Center, Yangzhou Data Bureau
- **Co-organizers:** Chinese Academy of Sciences Virtual Economy and Data Science Research Center, Jiangsu Big Data Alliance, Yangzhou Big Data Group, China Telecom Yangzhou Branch, China Mobile Yangzhou Branch, China Unicom Yangzhou Branch, Jiangsu Bank Yangzhou Branch, Yangzhou Big Data Group
- **Technical Support Unit:** Nanjing South Data Operations Research Institute

### **IV. Competition Tracks**

#### **Urban Governance Track**

With the acceleration of urbanization, city management faces unprecedented challenges. Issues such as street vending, garbage pile-up, and unlicensed peddlers pose higher demands on urban governance. This track focuses on intelligent detection of urban violations, requiring participants to develop efficient and reliable computer vision algorithms to enhance detection accuracy, reduce reliance on manual labor, and improve detection efficiency and effectiveness. The goal is to promote smarter and more efficient urban governance, creating a safer, more harmonious, and sustainable living environment for residents.

### **V. Competition Evaluation**

1. **Fairness and Impartiality:** Experts conduct online and offline evaluations while ensuring participant anonymity to maintain fairness. Competition organizers, operators, and supporting units involved in data access are prohibited from participating.
2. **Balanced Evaluation of Process and Outcome:** The competition assesses both the practicality and creativity demonstrated in the project development process.
3. **Theme Alignment:** Comprehensive evaluation of participants' understanding of the competition theme, in-depth data resource application, and overall performance in business analysis, model organization, technical solutions, feasibility analysis, cost considerations, and management factors.
4. **Evaluation Criteria:** The competition utilizes methods such as "work evaluation" and "final defense" to assess innovation ability, thinking skills, project value, and presentation skills.

## 3. Problem Statement

### **I. Description**

With the accelerating pace of urbanization, city governance faces unprecedented challenges and opportunities. The fine-grained and intelligent management of cities has become a key pursuit for excellence worldwide.

Illegal parking of motor vehicles, illegal parking of non-motor vehicles, and street vending are among the urban violations that disrupt aesthetics, order, and public well-being. Traditional manual inspections and passive response models are no longer sufficient for modern urban management needs.

This competition aims to develop an **intelligent recognition system** that can automatically detect and classify violations in urban management using advanced image processing and computer vision technologies. The system should analyze video footage captured by surveillance cameras, accurately identify violations, and promptly alert management authorities for more efficient urban governance.

### **II. Task Title**

**Intelligent Recognition of Urban Management Violations**

### **III. Task Details**

#### **Preliminary Round**

Participants must analyze urban surveillance video datasets to detect urban violations, including **overflowing garbage bins, illegal parking of motor vehicles, and illegal parking of non-motor vehicles**. The system must extract and mark violations with timestamps and location details.

#### **Preliminary Review**

- **Date:** September 13 - September 19
- **Top 24 teams advance to the next round. If any violations occur, alternate teams will be selected by the committee.**

#### **Semifinal Round**

Participants must analyze urban surveillance video datasets to detect urban violations, including **illegal business operations, overflowing garbage bins, illegal parking of motor vehicles, and illegal parking of non-motor vehicles**.

The semifinal dataset will be **unavailable to participants** for direct viewing. Participants will develop and submit models in a restricted environment for automated evaluation.

#### **Semifinal Review**

- **Date:** October 25 - November 15
- **Top 12 teams advance to the final. Any violations will lead to replacement by alternates.**

#### **Final Round**

The final evaluation will be conducted through an online or offline defense (TBD). Participants must prepare a presentation (PPT) and supporting materials. The final ranking will be based on performance in both previous rounds and the defense.

### **IV. Data Description**

- **Training Set (Labeled & Unlabeled)**
- **Test Set**
- **Data Format:** MP4 video files, JSON annotations

### **V. Evaluation Metrics**

#### **Preliminary Round**

- **F1 Score**
- **MOTA (Multiple Object Tracking Accuracy)**

#### **Semifinal Round**

- **Same accuracy metrics as Preliminary Round**
- **Efficiency evaluation using FPS (Frames Per Second)**

### **VI. Submission Instructions**

- Participants must generate a `result` folder containing JSON files corresponding to the videos.
- JSON format includes:
  - `frame_id`: Frame number of the violation
  - `event_id`: Violation ID
  - `category`: Violation category
  - `bbox`: Bounding box coordinates [xmin, ymin, xmax, ymax]
  - `confidence`: Detection confidence score

## 4. Project Repository

This project was developed on a cloud computing platform, and the repository provides only training code.

```python
├─Certification
│      2024_Grand_Canal_Cup.pdf
│
├─docs
│  │  1.Installation_Guide.md
│  │  2.Scoring_Criteria.md
│  │  3.YOLO.md
│  │  4.Object_Detection.md
│  └──5.Data_Augmentation.md
│
├─result
└─src
    │  train.py
    │
    └─yolo-dataset
        ├─train
        └─val
```

## 5. Project Execution

Please refer to [Cloud Platform Instructions](./docs/1.环境安装.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

```python
[1] Competition Link: https://www.marsbigdata.com/competition/details?id=3839107548872
[2] Object Detection Datasets: https://docs.ultralytics.com/datasets/detect/
[3] Performance Metrics Guide: https://docs.ultralytics.com/guides/yolo-performance-metrics/
[4]	YOLOv8:https://docs.ultralytics.com/models/yolov8/
[5]	https://docs.ultralytics.com/modes/predict/
[6]	https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results
[7]	https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/
```