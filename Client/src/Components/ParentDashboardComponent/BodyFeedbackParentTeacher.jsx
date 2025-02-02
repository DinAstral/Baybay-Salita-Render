import React, { useState, useEffect, useContext } from "react";
import axios from "axios";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import {
  Card,
  Button,
  Tooltip,
  CardHeader,
  CardBody,
  Spinner,
} from "@nextui-org/react";
import { UserContext } from "../../../context/userContext";
import toast from "react-hot-toast";

const BodyFeedbackParentTeacher = () => {
  const { user } = useContext(UserContext);
  const [feedbacks, setFeedbacks] = useState([]);
  const [parent, setParent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchParent = async () => {
      try {
        const response = await axios.get(`/api/getParent/${user.UserID}`);
        setParent(response.data);
      } catch (err) {
        toast.error("Failed to load parent data.");
      } finally {
        setLoading(false);
      }
    };
    fetchParent();
  }, [user.UserID]);

  useEffect(() => {
    const fetchFeedbacks = async () => {
      if (!parent?.Student?.[0]?.LRN) return;

      setLoading(true);
      try {
        const feedbackResponse = await axios.get(`/api/getFeedbacks`);
        const studentFeedbacks = feedbackResponse.data.filter(
          (feedback) => feedback.LRN === parent.Student[0].LRN
        );
        setFeedbacks(studentFeedbacks);
      } catch (error) {
        toast.error("Failed to load feedbacks. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchFeedbacks();
  }, [parent?.Student?.[0]?.LRN]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <Spinner size="lg" color="primary" />
      </div>
    );
  }

  return (
    <div className="px-4 md:px-9 py-2">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl md:text-3xl font-semibold">
            Student Feedback
          </h1>
          <Tooltip
            showArrow={true}
            content={
              <div className="p-2">
                <div className="text-sm font-bold">View Feedback</div>
                <div className="text-xs">
                  This function allows you to view the teachers' feedback in the
                  assessments.
                </div>
              </div>
            }
          >
            <FontAwesomeIcon icon={faCircleInfo} className="text-gray-600" />
          </Tooltip>
        </div>
      </div>

      <div className="content-container">
        {feedbacks.length > 0 ? (
          <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {feedbacks.map((feedback) => (
              <Card
                key={feedback._id}
                className="shadow-lg hover:shadow-2xl transition-shadow"
              >
                <CardHeader className="pb-0 pt-4 px-4 flex-col items-start">
                  <p className="text-lg uppercase font-bold text-primary">
                    {feedback.Title}
                  </p>
                  <h3 className="text-default-900">
                    Assessment Type: {feedback.Type}
                  </h3>
                  <h4 className="font-bold text-sm mt-1">
                    Assessment Code: {feedback.ActivityCode}
                  </h4>
                </CardHeader>
                <CardBody className="py-4">
                  <p className="mb-1">
                    <span className="font-semibold text-sm">
                      LRN: {feedback.LRN}
                    </span>
                  </p>
                  <p className="mb-1">
                    <span className="font-semibold text-sm">
                      Date: {feedback.Feedback_Date}
                    </span>
                  </p>
                  <p>
                    <span className="font-semibold text-sm">
                      Context: {feedback.Context}
                    </span>
                  </p>
                </CardBody>
              </Card>
            ))}
          </div>
        ) : (
          <p className="text-center text-gray-600 font-medium">
            No feedback available for this student.
          </p>
        )}
      </div>
    </div>
  );
};

export default BodyFeedbackParentTeacher;
