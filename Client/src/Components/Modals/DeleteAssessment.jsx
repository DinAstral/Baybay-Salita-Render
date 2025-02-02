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
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader>Delete Activity Successfully</ModalHeader>
        <ModalBody>
          <p>You have deleted the activity for the student successfully.</p>
        </ModalBody>
        <ModalFooter>
          <Button color="primary" onClick={onHide}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

const DeleteAssessment = ({ show, onHide, activity, onDeleteSuccess }) => {
  const [modalSubmitSuccess, setModalSubmitSuccess] = useState(false);

  const deleteActivity = async () => {
    try {
      await axios.delete(`/api/deleteAssessment/${activity._id}`);
      setModalSubmitSuccess(true);
      onDeleteSuccess();
    } catch (err) {
      console.error("Error deleting activity:", err);
      toast.error("Failed to delete activity. Please try again later.");
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
            Delete The Activity
          </ModalHeader>
          <ModalBody>
            <p>Do you want to delete the current activity?</p>
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
DeleteAssessment.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  activity: PropTypes.object, // Add activity prop type validation
};

export default DeleteAssessment;
