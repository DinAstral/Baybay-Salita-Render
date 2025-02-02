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

const UpdateProfile = ({ show, onHide, profile }) => {
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
          Update User Information
        </ModalHeader>
        <ModalBody>
          <p>
            You are now going to update your profile. Do you wish to continue?
          </p>
        </ModalBody>
        <ModalFooter>
          <Button variant="light" color="danger" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/AdminEditProfile/${profile}`}>
            <Button color="primary">Update</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

// Add prop types validation
UpdateProfile.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  profile: PropTypes.object,
};

export default UpdateProfile;
