import PropTypes from "prop-types"; // Import PropTypes
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";
import { Link } from "react-router-dom";

const EditUser = ({ show, onHide, user }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="edit-user-modal-title"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col" id="edit-user-modal-title">
          Add User Information
        </ModalHeader>
        <ModalBody>
          <p>
            You are now going to update the selected user. Do you wish to
            continue?
          </p>
        </ModalBody>
        <ModalFooter>
          <Button variant="light" color="danger" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/AdminEditUser/${user?.UserID}`}>
            <Button color="primary">Update</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

// Add prop types validation
EditUser.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  user: PropTypes.object,
};

export default EditUser;
