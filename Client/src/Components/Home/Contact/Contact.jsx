import React from "react";
import "./Contact.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faMessage,
  faEnvelope,
  faPhone,
  faLocationPin,
  faArrowRight,
} from "@fortawesome/free-solid-svg-icons";

const Contact = () => {
  const [result, setResult] = React.useState("");
  const onSubmit = async (event) => {
    event.preventDefault();
    setResult("Sending....");
    const formData = new FormData(event.target);

    formData.append("access_key", "458a4527-ac95-47b7-9cc2-f687b3fc2486");

    const response = await fetch("https://api.web3forms.com/submit", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      setResult("Form Submitted Successfully!");
      event.target.reset();
    } else {
      console.log("Error", data);
      setResult(data.message);
    }
  };
  return (
    <div className="contact">
      <div className="contact-col">
        <h3>
          Mag bigay ng iyong mensahe!{" "}
          <FontAwesomeIcon icon={faMessage} size="1x" className="message" />
        </h3>
        <p>
          Wag mahihiyang magtanong, kami ay inyong bigyan ng mensahe kung meron
          kayung gustong itanong sa amin. Ang inyong mga pahayag ay mahalaga
          samin at ito ay gagawin namin para mas mapaayus namin ang serbisyo na
          hatid namin sa inyo.
        </p>
        <ul>
          <li>
            <FontAwesomeIcon icon={faEnvelope} size="1x" className="img-icon" />{" "}
            baybaysalita@gmail.com
          </li>
          <li>
            <FontAwesomeIcon icon={faPhone} size="1x" className="img-icon" />{" "}
            +63 999 XXX XXXX
          </li>
          <li>
            <FontAwesomeIcon
              icon={faLocationPin}
              size="1x"
              className="img-icon"
            />{" "}
            National University Manila
          </li>
        </ul>
      </div>
      <div className="contact-col">
        <form onSubmit={onSubmit}>
          <label htmlFor="">Iyong Pangalan</label>
          <input
            type="text"
            name="name"
            placeholder="Ilagay ang iyong pangalan"
            required
          />
          <label htmlFor="">Telepono</label>
          <input
            type="tel"
            name="phone"
            placeholder="Ilagay ang iyong telepono"
            required
          />
          <label htmlFor="">Ang iyong mensahe</label>
          <textarea
            name="message"
            id="message"
            cols="30"
            rows="6"
            placeholder="Ilagay ang iyong mensahe"
            required
          ></textarea>
          <button type="submit" className="btn-home-send img">
            Ipasa{" "}
            <FontAwesomeIcon
              icon={faArrowRight}
              size="1x"
              className="message"
            />
          </button>
        </form>
        <span>{result}</span>
      </div>
    </div>
  );
};

export default Contact;
