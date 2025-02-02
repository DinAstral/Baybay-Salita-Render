/* eslint-disable react/prop-types */
/* eslint-disable react/no-unescaped-entities */
import { useState } from "react";
import PropTypes from "prop-types"; // Import PropTypes
import toast from "react-hot-toast";
import axios from "axios";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";

const DeleteSuccess = ({ show, onHide }) => (
  <Modal
    isOpen={show}
    onClose={onHide}
    aria-labelledby="contained-modal-title-vcenter"
    isDismissable={false}
    isKeyboardDismissDisabled={true}
  >
    <ModalContent>
      <ModalHeader id="contained-modal-title-vcenter">
        Delete User Successful
      </ModalHeader>
      <ModalBody>
        <p>You have successfully deleted the User's information.</p>
      </ModalBody>
      <ModalFooter>
        <Button color="danger" onClick={onHide}>
          Close
        </Button>
      </ModalFooter>
    </ModalContent>
  </Modal>
);

const DeleteUser = ({ show, onHide, user, onDeleteSuccess }) => {
  const [modalDeleteSuccess, setModalDeleteSuccess] = useState(false);

  const deleteUser = async () => {
    try {
      // Make sure user.email exists and is correct
      if (!user || !user.email) {
        toast.error("User not found.");
        return;
      }

      const response = await axios.delete(`/api/deleteUser/${user.email}`);
      if (response.status === 200) {
        // Check if the deletion was successful
        setModalDeleteSuccess(true);
        onDeleteSuccess();
      } else {
        toast.error("Failed to delete user. Please try again.");
      }
    } catch (err) {
      console.error("Error deleting user:", err);
      toast.error("Failed to delete user. Please try again later.");
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
          <ModalHeader id="contained-modal-title-vcenter">
            Delete User Information
          </ModalHeader>
          <ModalBody>
            <p>Are you sure you want to delete this User's information?</p>
          </ModalBody>
          <ModalFooter>
            <Button color="default" variant="light" onClick={onHide}>
              Cancel
            </Button>
            <Button color="danger" onClick={deleteUser}>
              Delete
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
      <DeleteSuccess
        show={modalDeleteSuccess}
        onHide={() => {
          setModalDeleteSuccess(false);
          onHide();
        }}
      />
    </>
  );
};

// Ensure you handle prop types
DeleteUser.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  onDeleteSuccess: PropTypes.func.isRequired,
  user: PropTypes.object, // Consider making this more specific if possible
};

export default DeleteUser;
