/* eslint-disable react/prop-types */
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

const DeleteActivitySuccess = ({ show, onHide }) => {
  const handleSuccessClick = () => {
    onHide();
    window.location.reload(); // You might want to change this to a more efficient state update instead of reloading the page.
  };

  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader>Delete Student Performance Successfully</ModalHeader>
        <ModalBody>
          <p>You have deleted the performance for the student successfully.</p>
        </ModalBody>
        <ModalFooter>
          <Button color="primary" onClick={handleSuccessClick}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

const DeletePerformance = ({
  show,
  onHide,
  performance,
  refreshActivities,
}) => {
  const [modalSubmitSuccess, setModalSubmitSuccess] = useState(false);

  const deleteActivity = async () => {
    try {
      await axios.delete(`/api/deletePerformance/${performance._id}`);
      setModalSubmitSuccess(true);
      refreshActivities(); // Automatically refresh activities
    } catch (err) {
      console.error("Error deleting performance:", err);
      toast.error("Failed to delete performance. Please try again later.");
    }
  };

  return (
    <>
      <Modal
        isOpen={show}
        onClose={onHide}
        aria-labelledby="contained-modal-title-vcenter"
        isDismissable={false}
        isKeyboardDismissDisabled={true}
      >
        <ModalContent>
          <ModalHeader className="flex flex-col">
            Delete The Student Performance
          </ModalHeader>
          <ModalBody>
            <p>Do you want to delete the current performance?</p>
          </ModalBody>
          <ModalFooter>
            <Button color="danger" variant="light" onClick={onHide}>
              Close
            </Button>
            <Button color="primary" onClick={deleteActivity}>
              Delete
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
      <DeleteActivitySuccess
        show={modalSubmitSuccess}
        onHide={() => {
          setModalSubmitSuccess(false);
          onHide();
        }}
      />
    </>
  );
};

// Add prop types validation
DeletePerformance.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  performance: PropTypes.object, // Add activity prop type validation
};

export default DeletePerformance;
