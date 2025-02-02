const _ = require("lodash");
const mongoose = require("mongoose");
const Performance = require("../models/performance");
const Student = require("../models/student");

const studentStatus = async (req, res) => {
  try {
    const { LRN } = req.params;

    // Fetch student by LRN
    const student = await Student.findOne({ LRN });
    if (!student) {
      return res.status(404).json({ error: "Student not found" });
    }

    const assessments = [
      { name: "Pagbabaybay", maxScore: 10 },
      { name: "Pantig", maxScore: 10 },
      { name: "Salita", maxScore: 10 },
      { name: "Pagbabasa", maxScore: 5 },
    ];

    const performances = await Performance.find({
      LRN: student.LRN,
      Type: { $in: assessments.map((a) => a.name) },
    });

    if (_.isEmpty(performances)) {
      const status = "Incomplete";
      const recommendations = [
        "The student must complete all assessments to get a full evaluation.",
        "Start with Pagbabaybay and Pantig for foundational skills.",
      ];
      await Student.findByIdAndUpdate(student._id, { status });

      return res.status(200).json({
        message: "Status updated to Incomplete.",
        status,
        recommendations,
      });
    }

    const totalScores = _.sumBy(performances, "Score");
    const totalPossibleScore = _.sumBy(assessments, "maxScore");
    const averageScore = (totalScores / totalPossibleScore) * 100;

    const completedAssessments = new Set(performances.map((p) => p.Type));
    const allAssessmentsCompleted = assessments.every((a) =>
      completedAssessments.has(a.name)
    );

    if (!allAssessmentsCompleted) {
      const status = "Incomplete";
      const recommendations = [
        "Identify and complete missing assessments.",
        "Encourage the student to complete all assessments for better insights.",
      ];
      await Student.findByIdAndUpdate(student._id, { status });

      return res.status(200).json({
        message: "Incomplete assessments detected.",
        status,
        recommendations,
      });
    }

    const levels = [
      {
        name: "Low Emerging Reader",
        min: 0,
        max: 16,
        comment: "Focus on phonics.",
        recs: ["Reinforce letter sounds."],
      },
      {
        name: "High Emerging Reader",
        min: 17,
        max: 30,
        comment: "Continue phonics work.",
        recs: ["Practice paired reading."],
      },
      {
        name: "Developing Reader",
        min: 31,
        max: 50,
        comment: "Work on syllables.",
        recs: ["Practice word blending."],
      },
      {
        name: "Transitioning Reader",
        min: 51,
        max: 75,
        comment: "Focus on comprehension.",
        recs: ["Add simple comprehension exercises."],
      },
      {
        name: "Grade Level Reader",
        min: 76,
        max: 100,
        comment: "Great reading fluency.",
        recs: ["Encourage genre exploration."],
      },
    ];

    const studentLevel = _.find(
      levels,
      (level) => averageScore >= level.min && averageScore <= level.max
    );
    const status = studentLevel.name;
    const comment = studentLevel.comment;

    await Student.findByIdAndUpdate(student._id, { status });

    // Additional Analysis: Determine words with frequent incorrect remarks
    const performanceItems = _.flatMap(performances, "PerformanceItems"); // Flatten PerformanceItems from all performances
    const incorrectRemarks = _.filter(
      performanceItems,
      (item) => item.Remarks && item.Remarks.toLowerCase() === "incorrect"
    );

    const wordErrorAnalysis = _(incorrectRemarks)
      .groupBy("Word")
      .map((items, word) => {
        const assessmentsWithIssues = _.uniq(
          items.map((item) => item.assessmentType)
        ); // Get unique assessment types
        return {
          word,
          totalAttempts: items.length,
          incorrectAttempts: items.length,
          incorrectPercentage: (items.length / performanceItems.length) * 100,
          assessmentsWithIssues,
        };
      })
      .orderBy("incorrectPercentage", "desc")
      .value();

    // Generate context-aware recommendations based on word errors and assessments
    const recommendations = wordErrorAnalysis.map((entry) => {
      const assessmentsStr = entry.assessmentsWithIssues.join(", ");
      return `The word "${entry.word}" has a high error rate in assessments: ${assessmentsStr}. Focus on practicing this word, especially in ${assessmentsStr}. Use phonics and syllable-blending activities to reinforce understanding.`;
    });

    // Return response with status, comment, context-aware recommendations, and word error analysis
    return res.status(200).json({
      message: `Status updated to ${status}`,
      totalScores,
      totalPossibleScore,
      averageScore,
      status,
      comment,
      recommendations,
      wordErrorAnalysis,
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Server error" });
  }
};

module.exports = { studentStatus };
