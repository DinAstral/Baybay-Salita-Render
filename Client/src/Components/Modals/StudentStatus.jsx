import { useState } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import PropTypes from "prop-types";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";

// Modal component to show the student's status, comments, and recommendations
const ShowStatus = ({
  show,
  onHide,
  status,
  comment,
  recommendations, // Accept recommendations as an array
}) => (
  <Modal
    isOpen={show}
    onClose={onHide}
    aria-labelledby="contained-modal-title-vcenter"
    isDismissable={false}
    isKeyboardDismissDisabled={true}
  >
    <ModalContent>
      <ModalHeader>Assess Student Performance Successfully</ModalHeader>
      <ModalBody>
        <p>
          <b>Status:</b> {status || "Status not available"}
        </p>
        <p>
          <b>Comment:</b> {comment || "No comments available"}
        </p>
        <div>
          <b>Recommendations:</b>
          {recommendations && recommendations.length > 0 ? (
            <ul>
              {recommendations.map((rec, index) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          ) : (
            <p>No recommendations available</p>
          )}
        </div>
      </ModalBody>
      <ModalFooter>
        <Button color="primary" onClick={onHide}>
          Close
        </Button>
      </ModalFooter>
    </ModalContent>
  </Modal>
);

const StudentStatus = ({ show, onHide, LRN, onStatusUpdate }) => {
  const [modalSubmitSuccess, setModalSubmitSuccess] = useState(false);
  const [status, setStatus] = useState("");
  const [comment, setComment] = useState("");
  const [recommendations, setRecommendations] = useState([]); // Update to store an array of recommendations

  // Handle the click event to assess the student's performance
  const handleAssessClick = async () => {
    try {
      const response = await axios.patch(`/api/studentStatus/${LRN}`);
      console.log("Response data:", response.data);

      const { status, comment, recommendations } = response.data; // Expect recommendations as an array
      onStatusUpdate(status);
      setStatus(status);
      setComment(comment);

      // Set recommendations as an array (or empty array if none available)
      setRecommendations(recommendations || []);

      setModalSubmitSuccess(true);
    } catch (error) {
      toast.error("Error updating student status");
      console.error("Error updating student status:", error);
    }
  };

  return (
    <>
      {/* Modal for initiating the assessment */}
      <Modal
        isOpen={show}
        onClose={onHide}
        aria-labelledby="contained-modal-title-vcenter"
        isDismissable={false}
        isKeyboardDismissDisabled={true}
      >
        <ModalContent>
          <ModalHeader className="flex flex-col">
            Assess Student Status
          </ModalHeader>
          <ModalBody>
            <p>
              Do you want to assess the performance of the selected student?
            </p>
          </ModalBody>
          <ModalFooter>
            <Button color="danger" variant="light" onClick={onHide}>
              Close
            </Button>
            <Button color="primary" onClick={handleAssessClick}>
              Assess
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* Modal for showing assessment results */}
      <ShowStatus
        show={modalSubmitSuccess}
        onHide={() => {
          setModalSubmitSuccess(false);
          onHide();
        }}
        status={status}
        comment={comment}
        recommendations={recommendations} // Pass the array of recommendations
      />
    </>
  );
};

// Prop types validation
StudentStatus.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  LRN: PropTypes.string.isRequired,
  onStatusUpdate: PropTypes.func.isRequired,
};

export default StudentStatus;
