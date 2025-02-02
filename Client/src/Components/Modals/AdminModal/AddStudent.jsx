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

const AddStudent = ({ show, onHide }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="add-user-modal-title"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col" id="add-user-modal-title">
          Add Student Information
        </ModalHeader>
        <ModalBody>
          <p>Do you want to add student information?</p>
        </ModalBody>
        <ModalFooter>
          <Button color="danger" variant="light" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/adminAddStudent`}>
            <Button color="primary">Add</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

// Add prop types validation
AddStudent.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
};

export default AddStudent;
