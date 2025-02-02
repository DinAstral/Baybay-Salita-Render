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

const ViewStudent = ({ show, onHide, student }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col" id="edit-user-modal-title">
          View Student Information
        </ModalHeader>
        <ModalBody>
          <p>Do you want to view the profile of student selected?</p>
        </ModalBody>
        <ModalFooter>
          <Button color="danger" variant="light" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/adminViewStudent/${student?._id}`}>
            <Button color="primary">View</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

ViewStudent.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  student: PropTypes.object,
};

export default ViewStudent;
